"""
Признаки мошенничества (RU/KZ): Evidence -> FeaturesResult (контракт из PROJECT_CONTEXT.md, раздел 5).

    {"flags":   {"has_telegram_link": 1, "guaranteed_income_phrase": 0, ...},   # всегда 0/1, все флаги из FLAG_NAMES
     "matches": [{"flag": "has_telegram_link", "source": "ocr", "t": 4.0, "text": "t.me/fastmoney"}]}

Зачем модуль нужен:
  1. Объяснимость: каждый флаг подтверждён цитатой и таймкодом (это и есть "улики" для вердикта).
  2. Вход для calibrate.py: флаги + скор LLM -> итоговый риск.
  3. Не зависит от OpenAI: работает мгновенно и бесплатно, поэтому годится для ablation.

Главная идея: слова про деньги ("инвестиции", "заработок", "акции") сами по себе НЕ признак скама,
их полно у честных финансовых блогеров (hard negatives). Поэтому флаги делятся на группы:
  contact   - как с тобой связываются (t.me, WhatsApp, "пиши в личку")
  claims    - обещания (гарантия, +300%, "без опыта", "это не развод")
  pressure  - давление и вербовка (срочность, "приведи друга", уровни/МЛМ, "подпишись и лайк")
  topic     - тема (беттинг, розыгрыш, вакансия, трейдинг) - слабые признаки
  counter   - признаки честного контента ("не гарантирую", "риск потерять", "как распознать мошенников")
  evasion   - обфускация ("инв3стиции", "з а р а б о т о к")

Использование:
    from src.features import extract_features
    res = extract_features(asr=asr_result, ocr=ocr_result, meta=ingest_result["meta"])
    # любой источник можно не передавать -> удобно для ablation:
    res_audio_only = extract_features(asr=asr_result)

Из консоли (читает data/cache/<video_id>.{asr,ocr,meta}.json, если есть):
    python -m src.features tt_7649544344751312150
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CACHE_DIR = Path("data/cache")
MAX_MATCHES_PER_FLAG_SOURCE = 3   # не больше 3 цитат на (флаг, источник), чтобы не раздувать JSON
SNIPPET_CONTEXT = 20              # символов контекста вокруг найденной фразы
SNIPPET_MAX = 140

# ----------------------------------------------------------------------------------------------
# Нормализация (длину строки НЕ меняет, поэтому цитаты берём из оригинала по тем же индексам)
# ----------------------------------------------------------------------------------------------
_CYR = "а-яәіңғүұқөһ"  # русские + казахские буквы (ё заменяем на е заранее)
_CYR_LETTER = re.compile(f"[{_CYR}]")
_LAT_LETTER = re.compile(r"[a-z]")
_TOKEN = re.compile(r"\w+")
# латинские двойники кириллицы: "зaрaботок" (латинские a), "кoнкурс"
_LAT_TO_CYR = str.maketrans({"a": "а", "c": "с", "e": "е", "k": "к", "m": "м", "o": "о",
                             "p": "р", "t": "т", "x": "х", "y": "у", "b": "в", "h": "н"})
# цифры вместо букв ("инв3стиции", "заработ0к"). Только ВНУТРИ слова между буквами,
# чтобы не портить "1000руб", "5х2", "2022г".
_DIGIT_TO_CYR = {"0": "о", "3": "е", "4": "ч", "6": "б"}
_SANDWICHED_DIGIT = re.compile(f"(?<=[{_CYR}a-z])[0346](?=[{_CYR}a-z])")
# побуквенное разбиение: "г а р а н т и я", "з.а.р.а.б.о.т.о.к"
_SPACED = re.compile(f"(?<!\\w)(?:[{_CYR}a-z][ .\\-_*]){{4,}}[{_CYR}a-z](?!\\w)")
_SPACED_SEP = re.compile(r"[ .\-_*]")


def _repair_token(tok: str) -> str:
    """Чинит слово, если в кириллице спрятаны латинские буквы или цифры. Иначе возвращает как есть."""
    if "_" in tok:                      # юзернеймы вроде trade_деньги - не обфускация
        return tok
    cyr = len(_CYR_LETTER.findall(tok))
    if cyr < 2:
        return tok
    lat = len(_LAT_LETTER.findall(tok))
    fixed = tok
    if lat and lat * 2 <= cyr:          # латиницы заметно меньше, чем кириллицы
        fixed = fixed.translate(_LAT_TO_CYR)
    fixed = _SANDWICHED_DIGIT.sub(lambda m: _DIGIT_TO_CYR[m.group()], fixed)
    return fixed


def normalize(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """lower + ё->е + исправление обфускации. Возвращает (текст, [(start, end) исправленных слов]).
    Длина результата равна длине text.lower()."""
    low = text.lower().replace("ё", "е")
    out, repairs, last = [], [], 0
    for m in _TOKEN.finditer(low):
        fixed = _repair_token(m.group())
        if fixed != m.group():
            repairs.append((m.start(), m.end()))
            out.append(low[last:m.start()])
            out.append(fixed)
            last = m.end()
    out.append(low[last:])
    return "".join(out), repairs


def despace(text: str) -> str:
    """'г а р а н т и я' -> 'гарантия' (склеивает только длинные цепочки одиночных букв)."""
    return _SPACED.sub(lambda m: _SPACED_SEP.sub("", m.group()), text)


# ----------------------------------------------------------------------------------------------
# Определения флагов: (имя, группа, [паттерны], negation_aware)
# Паттерн = строка-регулярка или (регулярка, минимум): тогда группа 1 должна быть числом >= минимума.
# Текст уже в нижнем регистре, ё -> е.
# ----------------------------------------------------------------------------------------------
_URLCHARS = r"[\w\-./?=&@%+]+"

_SPECS = [
    # ---------------- contact ----------------
    ("has_telegram_link", "contact", [
        rf"\bt\.me/{_URLCHARS}", rf"\btelegram\.(?:me|org|dog)/{_URLCHARS}"], False),
    ("has_whatsapp_link", "contact", [
        rf"\bwa\.me/{_URLCHARS}", rf"\b(?:chat|api)\.whatsapp\.com/{_URLCHARS}", r"\bwhatsapp\.com\b"], False),
    ("has_url", "contact", [
        rf"https?://{_URLCHARS}", rf"\bwww\.{_URLCHARS}", rf"\b(?:t|wa)\.me/{_URLCHARS}",
        rf"\b(?:bit\.ly|linktr\.ee|taplink\.cc)/{_URLCHARS}",
        r"\b[a-z0-9][a-z0-9\-]+\.(?:com|ru|kz|io|net|org|xyz|top|club|online|site|app|bio|cc|ly)\b"],
     False),
    ("has_phone_number", "contact", [
        r"(?<!\d)(?:\+?7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)",
        r"(?<!\d)\+\d{2,3}[\s\-()]*\d{2,3}[\s\-()]*\d{3}[\s\-]*\d{2,4}(?!\d)"], False),
    ("has_username_handle", "contact", [r"(?<![\w.])@[a-z0-9_.]{4,}"], False),
    ("contact_call_to_action", "contact", [
        r"пиши(?:те)?\s+(?:\w+\s+){0,3}?(?:в\s+)?(?:личк\w*|лс|директ\w*|telegram|телеграм\w*|тг|ватсап\w*|вацап\w*|whatsapp|direct|dm)\b",
        r"напиш\w+\s+(?:мне|в\s+личк\w*|в\s+директ\w*|слово|плюс|\+)",
        r"пиши(?:те)?\s+(?:мне|плюс\w*|\+)",
        r"ставь(?:те)?\s+(?:плюс\w*|\+)",
        r"в\s+комментари\w+\s+(?:плюс|\+|слово|напиш\w+)",
        r"\bdm\s+me\b|\b(?:message|text|write|contact)\s+me\b",
        r"\bжаз\w*\s+(?:маған|жеке|телеграм\w*|ватсап\w*)|\bжеке\s+(?:хабарлама\w*|чат\w*)|телеграм\w*\s*(?:ға|ге|қа|ке)?\s*жаз\w*",
    ], False),

    ("link_in_bio", "contact", [
        r"ссылк\w+\s+(?:в\s+)?(?:профил\w+|шапк\w+|описани\w+|bio|био)", r"переходи\w*\s+по\s+ссылк\w+",
        r"link\s+in\s+(?:my\s+)?(?:bio|description|profile)",
    ], False),

    # ---------------- claims ----------------
    ("guaranteed_income_phrase", "claims", [
        r"гарантир\w+", r"гарантированн\w+", r"гарантия\w*\s+(?:дохода|прибыл\w*|результат\w*|возврат\w*)",
        r"без\s+риск\w*", r"безрисков\w*", r"стопроцентн\w+", r"\bверняк\w*",
        r"\b100\s?%\s*(?:прибыл\w*|доход\w*|результат\w*|выигр\w*|прогноз\w*|проход\w*|заход\w*|гарант\w*)",
        r"\bкепілдік\w*|\bкепілдендір\w*|\bрисксіз\w*|\bрисксыз\w*|\bтәуекелсіз\w*",
        r"\bguarantee\w*", r"\brisk[\s\-]?free\b",
    ], True),
    ("unrealistic_return", "claims", [
        (r"(\d{2,4})\s?%\s*(?:в\s+|за\s+|на\s+|каждый\s+|каждую\s+|ежедневно\s+)?"
         r"(?:день|сутки|час|недел\w+|месяц\w*|\d+\s*(?:дн\w*|час\w*|недел\w*|мин\w*))", 20),
        (r"(?:\+|плюс\s*|до\s+)(\d{3,5})\s?%", 150),
        r"(?<!\w)[xх×]\s?[2-9](?!\w)", r"(?<!\w)[xх×]\s?\d{2,3}(?!\w)",
        r"удво\w+|утро\w+\s+(?:деньги|капитал|вложени\w+)|умнож\w+\s+(?:деньги|капитал|вложени\w+|доход\w*)|\bиксы\b|\bиксов\b",
        r"в\s+\d+\s+раз\w*\s+больше\s+(?:денег|доход\w*)",
    ], False),
    ("income_amount_claim", "claims", [
        r"\d[\d\s]*(?:тысяч\w*|тыс\b\.?|миллион\w*|млн|к)\s*(?:тенге|тг|₸|руб\w*|₽|долл\w*|бакс\w*|\$|usd|сом\w*)?\s+(?:в|за)\s+(?:день|неделю|месяц|сутки|час)",
        r"бір\s+(?:айда|күнде|аптада)\s+(?:\w+\s+){0,2}?\d",
        r"(?:заработ\w+|получ\w+|сделать|сделал\w*)\s+(?:\w+\s+){0,5}?за\s+(?:\d+|пару|несколько)\s+(?:минут\w*|час\w*|дн\w*)",
        r"\d[\d\s]*(?:долл\w*|бакс\w*|\$|usd)\s+(?:в|за)\s+(?:день|неделю|месяц|сутки)",
    ], False),
    ("easy_money_claim", "claims", [
        r"без\s+(?:опыта|вложени\w+|знани\w+|навыков|образовани\w+|специальн\w+)",
        r"опыт\w*\s+не\s+(?:нужен|требуется|обязателен)", r"не\s+нужн\w+\s+(?:опыт|знани\w+|вложени\w+)",
        r"(?:любой|каждый)\s+(?:новичок|человек)\s+(?:справится|сможет|может)", r"справится\s+(?:абсолютно\s+)?любой",
        r"\bлегк\w+\s+(?:деньги|заработок|работа|доход)", r"быстр\w+\s+(?:деньги|заработок|доход|возврат)",
        r"деньги\s+(?:сами|на\s+автомате)", r"\beasy\s+money\b|\bno\s+experience\b|\bpassive\s+income\b",
        r"(?:по|всего)\s+(?:\d+|час|два|пару|пол\s*часа)\s*(?:часа?|часов|минут\w*)\s+в\s+день",
        r"не\s+выходя\s+из\s+дома", r"возраст\s+от\s+\d+",
        r"тәжірибе\s+(?:қажет\s+емес|керек\s+емес)", r"тәжірибесіз", r"оңай\s+(?:табыс|ақша|жұмыс)", r"үйден\s+шықпай",
    ], False),
    ("income_boast", "claims", [
        r"\bя\s+(?:уже\s+)?(?:заработал\w*|сделал\w*|вывел\w*|получил\w*|поднял\w*)\s+(?:\w+\s+){0,3}?\d",
        r"(?:мои|мой)\s+(?:ученик\w*|подписчик\w*|студент\w*)\s+(?:заработал\w*|зарабатыва\w+|выводя\w+)",
        r"(?:закрыл\w*|закрыли)\s+(?:\w+\s+){0,2}?сделк\w*\s+(?:в\s+)?плюс",
        r"\bвывел\w*\s+(?:\w+\s+){0,3}?\d", r"\bвывел\w*\s+(?:первые|первую|свои|мои|деньги|прибыль)",
        r"\b(?:мен|біз)\s+(?:\w+\s+){0,4}?(?:табыс|табыз)\w*",
    ], False),
    ("legitimacy_claim", "claims", [
        r"это\s+не\s+(?:развод\w*|обман\w*|лохотрон\w*|пирамид\w+|шутк\w+|скам\w*|мошеннич\w+)",
        r"\bне\s+(?:развод|обман|лохотрон)\b", r"без\s+обмана",
        r"(?:все|всё)\s+(?:легальн\w*|прозрачн\w*)", r"(?:проект|компания)\s+(?:официально\s+)?зарегистрирован\w*",
        r"официальн\w+\s+зарегистрирован\w*", r"реальн\w+\s+(?:деньги|доход\w*|заработок)",
        r"\b(?:мен|біз)\s+алаяқ\w*\s+емес", r"алдау\s+емес",
    ], False),
    ("upfront_payment_phrase", "claims", [
        r"предоплат\w*", r"вступительн\w+\s+взнос\w*", r"стартов\w+\s+(?:пакет\w*|взнос\w*|набор\w*|капитал)",
        r"регистрационн\w+\s+взнос\w*", r"(?:внес\w+|внеси\w*|оставь\w*|оплат\w+)\s+(?:\w+\s+){0,2}?(?:залог|взнос|сумму|депозит)",
        r"оплатить\s+(?:обучение|доступ|курс|пакет|регистрацию)",
        r"купи\w*\s+(?:пакет|набор|стартов\w+|лицензи\w+|доступ)",
        r"минимальн\w+\s+(?:депозит|вклад|взнос)", r"пополни\w*\s+(?:баланс|счет|кошел\w+)",
        r"депозит\w*\s+(?:от\s+)?\d", r"вложи\w*\s+(?:от\s+)?\d",
        r"от\s+\d[\d\s]*\s*(?:долларов|баксов|тенге|\$|usd|руб\w*)\s+(?:на\s+(?:старт|начал\w+)|для\s+(?:старт\w+|начал\w+|входа))",
        r"алғашқы\s+(?:салым|төлем)",
    ], False),

    # ---------------- pressure ----------------
    ("urgency_words", "pressure", [
        r"\bсрочно\b", r"только\s+(?:сегодня|сейчас)", r"ограничен\w+\s+(?:количеств\w*|число|мест\w*|предложени\w*|время|акци\w*)",
        r"осталось\s+(?:всего\s+)?(?:\d+|несколько|пару)\s+(?:мест\w*|слот\w*|штук|человек\w*)",
        r"\bуспей\w*|\bуспейте\b|торопи\w+|поторопи\w+", r"последн\w+\s+(?:шанс|мест\w*|день)",
        r"пока\s+(?:не\s+)?(?:закрыл\w*|не\s+поздно|есть\s+места)", r"не\s+упусти\w*",
        r"(?:сегодня|завтра)\s+(?:последний|заканчива\w+)", r"в\s+течение\s+\d+\s+(?:минут\w*|часов|часа|сек\w*)",
        r"(?:у\s+тебя|у\s+вас)\s+есть\s+\d+\s+(?:секунд\w*|минут\w*)",
        r"\bтек\s+бүгін\b|\bүлгер\w*|\bасық\w*|\bшектеулі\b",
    ], False),
    ("referral_phrase", "pressure", [
        r"приведи\w*\s+(?:друг\w*|друзей|знаком\w+|людей)", r"пригла(?:си\w*|шай\w*)\s+(?:друг\w*|друзей|знаком\w+|людей|других)",
        r"добав(?:ь|ьте|ляй\w*)\s+(?:\w+\s+){0,3}?(?:друзей|друга|людей|знаком\w+)",
        r"реферал\w*|\bрефк\w*", r"за\s+каждого\s+(?:приглашенн\w+|друга|нового|привед\w+)",
        r"процент\w*\s+(?:от\s+)?(?:приглашенн\w+|привед\w+|команд\w+|структур\w+)",
        r"(?:твоя|ваша|своя|свою)\s+(?:команд\w+|структур\w+|сеть|линия|линию)",
        r"привлекай\w*|привлекать\s+(?:людей|партнер\w*|новых)", r"новых\s+(?:участник\w+|партнер\w+)",
        r"достарыңызды\s+шақыр\w*|шақыр\w+\s+(?:достар\w*|адам\w*)|\bрефералд\w*",
        r"\b(?:add|invite|bring)\s+(?:\w+\s+){0,2}?friends\b", r"\breferral\w*|\bdownline\b",
    ], False),
    ("mlm_structure_terms", "pressure", [
        r"многоуровнев\w+", r"сетев\w+\s+маркетинг\w*", r"\bмлм\b", r"\bmlm\b", r"\bматриц\w+",
        r"денежн\w+\s+маркетинг\w*", r"маркетинг[\s\-]*план\w*", r"\bуров(?:ень|ня|ни|ней|не)\b", r"ступен\w+",
        r"единоразов\w+\s+вложени\w+", r"возврат\s+вложенн\w+", r"партн[еe]рск\w+\s+(?:программ\w+|сет\w+)",
        r"\bvip\s?\d+\b", r"\blevels?\s+(?:vip|\d)", r"\bnetwork\s+marketing\b|\bmatrix\b",
    ], False),
    ("follow_like_share", "pressure", [
        r"подпиш\w+[^.!?]{0,80}?(?:лайк\w*|репост\w*|коммент\w+|друзей|друга)",
        r"(?:лайк\w*|репост\w*)[^.!?]{0,80}?подпи\w+",
        r"(?:отмет\w+|тегн\w+)\s+(?:\w+\s+){0,2}(?:друг\w*|друзей)", r"постав\w+\s+лайк", r"сделай\w*\s+репост",
        r"жазыл\w*[^.!?]{0,60}?(?:лайк|репост)",
    ], False),

    # ---------------- topic ----------------
    ("betting_terms", "topic", [
        r"\bставк\w*\s+на\s+(?:спорт|матч|футбол|хоккей|теннис|баскетбол|кибер\w*|игр\w+|исход\w*|побед\w+|команд\w+|тотал|экспресс|лайв)",
        r"(?:делать|делай\w*|сделать|сделал\w*|ставить|ставь(?:те)?|поставить|поставил\w*)\s+(?:\w+\s+){0,2}?(?:ставк\w+|на\s+(?:спорт|матч|футбол|исход|побед\w+|экспресс|команд\w+|фаворит\w*))",
        r"\bставки\s+на\b|букмекер\w*|\bбк\b|\bкэф\w*|\bкоэф\w*\s+(?:\d|на\s+(?:побед\w+|матч|исход))|экспресс\w*|фрибет\w*",
        r"\bbookmaker\w*|\bsportsbook\w*|\bparlay\b|\bbetting\s+(?:tips|odds|picks|history)\b|\bpaddy\s+power\b|\bbetfair\b",
        r"прогноз\w*\s+на\s+(?:матч|спорт|игр\w+|футбол)", r"проходимост\w+", r"\bтотал\b", r"\bфора\b",
        r"1xbet|1хбет|мелбет|melbet|1win|mostbet|мостбет|pin[\s\-]?up|пинап|париматч|parimatch|fonbet|фонбет|betboom|бетбум|winline|винлайн|лига\s+ставок",
        r"казино|\bслот\w*|рулетк\w*|тотализатор", r"\bбәс\w*", r"ке(?:ш|сш)[\s\-]?бе\w*|кэш[\s\-]?бэ\w*",
    ], False),
    ("promo_code_mention", "topic", [r"промо[\s\-]?код\w*|\bпромик\w*|promo[\s\-]?code"], False),
    ("giveaway_terms", "topic", [
        r"розыгрыш\w*|разыгр\w+|раздач\w+|раздаю|раздаем|раздадим", r"выигра\w+|выиграй\w*|выиграть",
        r"\bприз(?:ы|ов|а|у|ом|е)?\b", r"победител\w+", r"\bподар(?:ок|ки|ков|ку|ком)\b|\bподарк\w+",
        r"айфон\w*|iphone\w*|\bконкурс\w*|лотере\w+", r"ұтыс\w*|ұтып\w*|сыйлық\w*",
        r"\bgiveaway\b|\bwin\s+(?:an?\s+)?(?:iphone|prize|cash)\b",
    ], False),
    ("job_offer_terms", "topic", [
        r"работ\w+\s+(?:на\s+дому|удаленн\w+|из\s+дома|онлайн)", r"удаленн\w+\s+(?:работ\w+|сотрудник\w*|занятость)",
        r"подработк\w*", r"вакан\w+",
        r"набира\w+\s+(?:людей|сотрудник\w*|команду|в\s+команду|персонал|девушек|парней)",
        r"нужн\w+\s+(?:сотрудник\w*|люди|менеджер\w*|помощник\w*|администратор\w*|оператор\w*)",
        r"зарплат\w+\s+на\s+карт\w+", r"(?:свободн|удобн|гибк)\w+\s+график", r"график\s+(?:свободный|удобный|гибкий)",
        r"тайн\w+\s+покупател\w+", r"(?:обработка|заполнение)\s+(?:заявок|анкет|данных)",
        r"жұмыс\w*\s+(?:керек|қажет|іздеп|үйден)", r"үйден\s+жұмыс|ақылы\s+жұмыс",
    ], False),
    ("trading_terms", "topic", [
        r"инвест\w+", r"трейд\w+", r"криптовалют\w*|\bкрипт\w+|\busdt\b|биткоин\w*|\bbtc\b", r"форекс|\bforex\b",
        r"бинарн\w+\s+опцион\w*|\bопцион\w*", r"брокер\w*", r"торгов\w+\s+(?:бот\w*|робот\w*|сигнал\w*|стратег\w*)",
        r"сигнал\w*", r"копи[\s\-]?трейд\w*", r"\bбот\b", r"\bсделк\w+", r"\bдепозит\w*", r"\bиндикатор\w*",
        r"метатрейдер|\bmt[45]\b", r"pocket\s?option|binomo|биномо|olymp\s?trade|iq\s?option|quotex|бинариум|\bbybit\b|\bbinance\b",
        r"пассивн\w+\s+(?:доход|заработок)", r"\binvest\w*|\btrading\b|\bcrypto\w*", r"\bакци\w+|дивиденд\w*", r"\bвклад\w*|вложени\w+|\bвложи\w+|салым\w*",
    ], False),

    # ---------------- counter (признаки честного контента) ----------------
    ("risk_disclaimer", "counter", [
        r"не\s+гарантир\w+", r"никто\s+не\s+гарантир\w+", r"нет\s+гаранти\w+", r"гаранти\w+\s+нет", r"без\s+гаранти\w+",
        r"не\s+является\s+(?:финансов\w+|инвестиционн\w+)\s+(?:совет\w*|рекомендаци\w*)",
        r"не\s+(?:финансов\w+|инвестиционн\w+)\s+совет", r"можно\s+(?:потерять|остаться\s+без)",
        r"рискуеш\w+|рискуете", r"риск\w*\s+(?:потер\w+|убытк\w+)", r"высок\w+\s+риск\w*",
        r"потерять\s+(?:все|всё|деньги|вложени\w+)", r"убытк\w+", r"диверсифиц\w+",
        r"не\s+бывает\s+(?:гарантий|легких|быстрых)", r"\bdyor\b|\bnfa\b",
        r"кепілдік\s+жоқ|тәуекел\w*\s+(?:бар|жоғары)",
    ], False),
    ("anti_scam_context", "counter", [
        r"мошенник\w*|мошеннич\w+|обманыва\w+|обманут\w*|\bобман\w*|\bразвод\w*|лохотрон\w*",
        r"остерег\w+|осторожн\w+|не\s+ведись|не\s+ведитесь|как\s+(?:распознать|проверить|не\s+попасть)",
        r"признак\w+\s+(?:финансов\w+\s+)?пирамид\w*", r"\bscam\w*", r"алаяқ\w*|алдау\w*",
    ], True),
    ("educational_phrase", "counter", [
        r"для\s+(?:начинающих|новичков|чайников)", r"финансов\w+\s+грамотност\w+",
        r"как\s+(?:составить|выбрать|начать|накопить|правильно|распределить|откладывать)", r"что\s+такое",
        r"(?:\d+|пять|три|топ)\s+(?:правил\w*|совет\w*|способ\w*|идей|ошиб\w+|вещей|приложений)",
        r"давайте\s+разбер\w+",
    ], False),
    # obfuscation_detected заполняется отдельно (из normalize/despace), паттернов у него нет
    ("obfuscation_detected", "evasion", [], False),
]

FLAG_NAMES: Tuple[str, ...] = tuple(s[0] for s in _SPECS)
FLAG_GROUPS: Dict[str, str] = {s[0]: s[1] for s in _SPECS}

FLAG_DOCS: Dict[str, str] = {
    "has_telegram_link": "ссылка t.me / telegram.me",
    "has_whatsapp_link": "ссылка wa.me / whatsapp.com",
    "has_url": "любая внешняя ссылка",
    "has_phone_number": "номер телефона",
    "has_username_handle": "@юзернейм для связи",
    "contact_call_to_action": "«пиши в личку / ставь плюс / напишите мне»",
    "link_in_bio": "«ссылка в шапке профиля» (слабый: есть и у честных блогеров)",
    "guaranteed_income_phrase": "гарантия дохода / «без риска» / «100% прогноз» (без отрицания)",
    "unrealistic_return": "+300%, 30% в неделю, x2, «удвоим»",
    "income_amount_claim": "«N тысяч в день/неделю/месяц»",
    "easy_money_claim": "«без опыта», «лёгкие деньги», «не выходя из дома»",
    "income_boast": "«я заработал N», «мои ученики заработали»",
    "legitimacy_claim": "«это не развод», «всё легально, официально зарегистрирован»",
    "upfront_payment_phrase": "предоплата, стартовый пакет, минимальный депозит",
    "urgency_words": "срочность, «осталось 3 места», «успей»",
    "referral_phrase": "«приведи друга», рефералка, «моя команда»",
    "mlm_structure_terms": "уровни, матрица, сетевой маркетинг",
    "follow_like_share": "«подпишись + лайк + репост»",
    "betting_terms": "ставки, букмекер, кэф, казино, бренды БК",
    "promo_code_mention": "промокод",
    "giveaway_terms": "розыгрыш, приз, выиграть, айфон",
    "job_offer_terms": "работа на дому, подработка, набираю людей",
    "trading_terms": "инвестиции, трейдинг, крипта, депозит (слабый признак)",
    "risk_disclaimer": "«не гарантирую», «можно потерять деньги» - признак честного контента",
    "anti_scam_context": "предупреждение о мошенниках - признак честного контента",
    "educational_phrase": "«для начинающих», «5 правил» - образовательный тон",
    "obfuscation_detected": "«инв3стиции», «з а р а б о т о к» - попытка обойти фильтр",
}

_NEG_BEFORE = {"не", "нет", "ни", "нельзя", "никто", "никогда", "без", "жоқ", "емес",
               "not", "no", "never", "dont", "don", "cannot", "cant", "without"}
_NEG_AFTER = {"нет", "жоқ", "емес"}


def _compile_specs():
    compiled = []
    for name, group, patterns, neg in _SPECS:
        pats = []
        for p in patterns:
            if isinstance(p, tuple):
                pats.append((re.compile(p[0]), p[1]))
            else:
                pats.append((re.compile(p), None))
        compiled.append((name, group, pats, neg))
    return compiled


_COMPILED = _compile_specs()


def _negated(text: str, s: int, e: int) -> bool:
    """Фраза стоит в отрицании: «никто не гарантирует», «гарантий нет», «не обман»."""
    before = re.findall(r"\w+", text[max(0, s - 40):s])[-3:]
    if any(w in _NEG_BEFORE for w in before):
        return True
    after = re.findall(r"\w+", text[e:e + 30])[:2]
    return any(w in _NEG_AFTER for w in after) or after == ["не", "бывает"]


def _find_spans(spec, norm: str) -> List[Tuple[int, int]]:
    _name, _group, pats, neg = spec
    spans = []
    for rx, minimum in pats:
        for m in rx.finditer(norm):
            if minimum is not None:
                try:
                    if int(m.group(1)) < minimum:
                        continue
                except (ValueError, IndexError):
                    continue
            if neg and _negated(norm, m.start(), m.end()):
                continue
            spans.append((m.start(), m.end()))
    return spans


def _snippet(display: str, s: int, e: int) -> str:
    a, b = max(0, s - SNIPPET_CONTEXT), min(len(display), e + SNIPPET_CONTEXT)
    return " ".join(display[a:b].split())[:SNIPPET_MAX]


# ----------------------------------------------------------------------------------------------
# Сканирование одного текста
# ----------------------------------------------------------------------------------------------
def _scan(text: str) -> List[Tuple[str, int, int, str]]:
    """Возвращает [(flag, start, end, snippet)]. Позиции относятся к строке text (длина сохраняется)."""
    if not text or not text.strip():
        return []
    norm, repairs = normalize(text)
    display = text if len(text) == len(norm) else norm  # lower() редко меняет длину; тогда цитируем нормализованное
    hits = []
    for spec in _COMPILED:
        if not spec[2]:
            continue
        for s, e in _find_spans(spec, norm):
            hits.append((spec[0], s, e, _snippet(display, s, e)))
    for s, e in repairs:
        hits.append(("obfuscation_detected", s, e, _snippet(display, s, e)))
    return hits


def _scan_despaced(text: str) -> List[Tuple[str, str]]:
    """Второй проход для текста с побуквенным разбиением. Возвращает [(flag, snippet)] (без позиций)."""
    if not text:
        return []
    norm, _ = normalize(text)
    if not _SPACED.search(norm):
        return []
    joined = despace(norm)
    hits = [("obfuscation_detected", " ".join(m.group().split())[:SNIPPET_MAX]) for m in _SPACED.finditer(norm)]
    for spec in _COMPILED:
        if not spec[2]:
            continue
        for s, e in _find_spans(spec, joined):
            hits.append((spec[0], _snippet(joined, s, e)))
    return hits


# ----------------------------------------------------------------------------------------------
# Публичный API
# ----------------------------------------------------------------------------------------------
def _parts_from_ocr(ocr) -> List[Tuple[Optional[float], str]]:
    items = ocr.get("items", []) if isinstance(ocr, dict) else (ocr or [])
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append((it.get("t", it.get("timestamp")), str(it.get("text", ""))))
    return out


def _parts_from_meta(meta) -> List[Tuple[Optional[float], str]]:
    meta = meta or {}
    parts = [(None, str(meta.get("description") or "")),
             (None, " ".join("#" + str(h).lstrip("#") for h in (meta.get("hashtags") or []))),
             (None, str(meta.get("author_bio") or ""))]
    return [p for p in parts if p[1].strip()]


def extract_features(asr: Optional[dict] = None, ocr=None, meta: Optional[dict] = None) -> dict:
    """Evidence -> FeaturesResult.

    asr  - AsrResult {"language", "text", "segments": [{"start", "end", "text"}]}
    ocr  - OcrResult {"items": [{"t", "text", "confidence"}]} (допустим и просто список items)
    meta - ingest["meta"] {"description", "hashtags", "author", "author_bio"}
    Любой источник можно опустить (None) - так делается ablation «только речь / только экран / только текст».
    """
    flags = {name: 0 for name in FLAG_NAMES}
    matches: List[dict] = []
    seen = set()
    counts: Dict[Tuple[str, str], int] = {}

    def add(flag: str, source: str, t, text: str):
        flags[flag] = 1
        key = (flag, source, text)
        if key in seen or counts.get((flag, source), 0) >= MAX_MATCHES_PER_FLAG_SOURCE:
            return
        seen.add(key)
        counts[(flag, source)] = counts.get((flag, source), 0) + 1
        matches.append({"flag": flag, "source": source, "t": t, "text": text})

    # --- речь: сканируем ВЕСЬ транскрипт одним куском, чтобы ловить фразы на стыке сегментов ---
    asr = asr or {}
    segs = [s for s in (asr.get("segments") or []) if str(s.get("text", "")).strip()]
    if segs:
        texts, starts, pos = [], [], 0
        for s in segs:
            starts.append(pos)
            texts.append(str(s["text"]).strip())
            pos += len(texts[-1]) + 1
        joined = " ".join(texts)

        def seg_t(offset: int):
            idx = 0
            for i, st in enumerate(starts):
                if st <= offset:
                    idx = i
                else:
                    break
            return segs[idx].get("start")

        for flag, s, _e, snip in _scan(joined):
            add(flag, "asr", seg_t(s), snip)
        for seg, text in zip(segs, texts):
            for flag, snip in _scan_despaced(text):
                add(flag, "asr", seg.get("start"), snip)
    elif str(asr.get("text") or "").strip():
        for flag, _s, _e, snip in _scan(str(asr["text"])):
            add(flag, "asr", None, snip)

    # --- текст на экране и метаданные: каждый кусок отдельно ---
    for source, parts in (("ocr", _parts_from_ocr(ocr)), ("meta", _parts_from_meta(meta))):
        for t, text in parts:
            for flag, _s, _e, snip in _scan(text):
                add(flag, source, t, snip)
            for flag, snip in _scan_despaced(text):
                add(flag, source, t, snip)

    return {"flags": flags, "matches": matches}


def feature_vector(result: dict, names: Iterable[str] = FLAG_NAMES) -> List[int]:
    """FeaturesResult -> список 0/1 в фиксированном порядке (для sklearn)."""
    return [int(result["flags"].get(n, 0)) for n in names]


# ----------------------------------------------------------------------------------------------
# CLI: python -m src.features <video_id>
# ----------------------------------------------------------------------------------------------
def load_cached(video_id: str) -> dict:
    out = {}
    for kind in ("asr", "ocr", "meta"):
        f = CACHE_DIR / f"{video_id}.{kind}.json"
        out[kind] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Использование: python -m src.features <video_id>   (например tt_7649544344751312150)")
    print(json.dumps(extract_features(**load_cached(sys.argv[1])), ensure_ascii=False, indent=2))
