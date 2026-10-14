"""Тесты src/features.py. Запуск из корня репозитория:  python -m pytest tests/test_features.py -v"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FLAG_NAMES, extract_features, feature_vector  # noqa: E402


def asr(*pairs):
    """asr((4.2, 'текст'), ...) -> AsrResult"""
    segs = [{"start": t, "end": t + 3, "text": x} for t, x in pairs]
    return {"language": "ru", "text": " ".join(x for _, x in pairs), "segments": segs}


def flags_of(**kw):
    return extract_features(**kw)["flags"]


# ---------- контракт ----------
def test_contract_shape_and_types():
    res = extract_features(asr=asr((11.2, "гарантирую доход сорок процентов")),
                           ocr={"items": [{"t": 4.0, "text": "t.me/fastmoney  +300%", "confidence": 0.9}]},
                           meta={"description": "x", "hashtags": ["заработок"], "author": "u", "author_bio": ""})
    assert set(res) == {"flags", "matches"}
    assert tuple(res["flags"]) == FLAG_NAMES                      # все флаги, в фиксированном порядке
    assert all(v in (0, 1) and isinstance(v, int) for v in res["flags"].values())
    for m in res["matches"]:
        assert set(m) == {"flag", "source", "t", "text"}
        assert m["source"] in {"asr", "ocr", "meta"} and res["flags"][m["flag"]] == 1
    json.dumps(res, ensure_ascii=False)                           # сериализуется
    assert len(feature_vector(res)) == len(FLAG_NAMES)


def test_empty_inputs_are_safe():
    for kw in ({}, {"asr": None}, {"asr": {"language": None, "text": "", "segments": []}},
               {"ocr": {"items": []}}, {"ocr": []}, {"meta": {}}, {"meta": {"hashtags": None}}):
        res = extract_features(**kw)
        assert sum(res["flags"].values()) == 0 and res["matches"] == []


def test_ocr_accepts_list_and_dict():
    item = [{"t": 2, "text": "пиши в личку"}]
    assert flags_of(ocr=item)["contact_call_to_action"] == 1
    assert flags_of(ocr={"items": item})["contact_call_to_action"] == 1


# ---------- улики с таймкодами ----------
def test_ocr_telegram_link_has_timecode_and_quote():
    res = extract_features(ocr={"items": [{"t": 4.0, "text": "t.me/fastmoney  +300%", "confidence": 0.91}]})
    m = [x for x in res["matches"] if x["flag"] == "has_telegram_link"][0]
    assert m["source"] == "ocr" and m["t"] == 4.0 and "t.me/fastmoney" in m["text"]
    assert res["flags"]["unrealistic_return"] == 1                 # +300%


def test_asr_timecode_points_to_right_segment():
    res = extract_features(asr=asr((0.0, "всем привет сегодня расскажу"), (11.2, "гарантирую доход сорок процентов")))
    m = [x for x in res["matches"] if x["flag"] == "guaranteed_income_phrase"][0]
    assert m["t"] == 11.2 and m["source"] == "asr"


def test_phrase_across_segment_boundary():
    res = extract_features(asr=asr((1.0, "подпишись на канал и"), (4.0, "поставь лайк под видео")))
    assert res["flags"]["follow_like_share"] == 1


def test_meta_sources():
    res = extract_features(meta={"description": "Работа на дому без опыта", "hashtags": ["#заработок"],
                                 "author": "u", "author_bio": "пиши в личку t.me/work_kz"})
    assert res["flags"]["job_offer_terms"] == 1 and res["flags"]["has_telegram_link"] == 1
    assert all(m["source"] == "meta" and m["t"] is None for m in res["matches"])


# ---------- отрицание: главное против ложных срабатываний ----------
def test_negated_guarantee_is_disclaimer_not_scam():
    for phrase in ("никто не гарантирует доход на бирже", "гарантий нет, можно потерять деньги", "я не гарантирую прибыль"):
        f = flags_of(asr=asr((0, phrase)))
        assert f["guaranteed_income_phrase"] == 0, phrase
        assert f["risk_disclaimer"] == 1, phrase


def test_positive_guarantee_fires():
    for phrase in ("гарантирую доход сорок процентов", "это торговля без риска", "100% прогноз на матч", "кэф гарантирован"):
        assert flags_of(asr=asr((0, phrase)))["guaranteed_income_phrase"] == 1, phrase


def test_anti_scam_is_not_scam_but_denial_is():
    warn = flags_of(asr=asr((0, "осторожно, мошенники обманывают людей, как распознать пирамиду")))
    assert warn["anti_scam_context"] == 1
    denial = flags_of(asr=asr((0, "это не развод и не обман, проект официально зарегистрирован, все легально")))
    assert denial["legitimacy_claim"] == 1 and denial["anti_scam_context"] == 0   # «не обман» != предупреждение


# ---------- обфускация ----------
def test_digit_and_latin_homoglyphs():
    r = extract_features(asr=asr((0, "лучшие инв3стиции и зaрaботок без вложений")))   # 3 вместо е, латинские a
    assert r["flags"]["obfuscation_detected"] == 1 and r["flags"]["trading_terms"] == 1


def test_spaced_letters():
    r = extract_features(asr=asr((0, "я г а р а н т и р у ю тебе доход")))
    assert r["flags"]["obfuscation_detected"] == 1 and r["flags"]["guaranteed_income_phrase"] == 1


def test_no_obfuscation_false_positives():
    text = "Заплатил 1000руб за 5х2 график, смотри t.me/trade_деньги и iPhone15, 3-х комнатная, и т. д. 2022г"
    assert flags_of(asr=asr((0, text)))["obfuscation_detected"] == 0


# ---------- казахский ----------
def test_kazakh_markers():
    f = flags_of(asr=asr((0, "Бір айда 3 миллион тенге таза табыс, рисксіз, тәжірибе қажет емес")))
    assert f["guaranteed_income_phrase"] == 1 and f["income_amount_claim"] == 1 and f["easy_money_claim"] == 1


# ---------- hard negatives: честный финансовый контент ----------
def test_honest_finance_blogger_has_no_scam_claims():
    text = ("Как составить личный финансовый план? Это стратегия управления доходами, расходами, сбережениями "
            "и инвестициями. Помни, что инвестиции — это риск, никто не гарантирует доходность. Это не финансовый совет.")
    f = flags_of(asr=asr((0, text)))
    for scam_flag in ("guaranteed_income_phrase", "unrealistic_return", "urgency_words", "referral_phrase",
                      "contact_call_to_action", "legitimacy_claim", "upfront_payment_phrase", "easy_money_claim"):
        assert f[scam_flag] == 0, scam_flag
    assert f["trading_terms"] == 1 and f["risk_disclaimer"] == 1 and f["educational_phrase"] == 1


def test_word_money_alone_is_not_a_scam():
    f = flags_of(asr=asr((0, "Деньги любят счёт. Я откладываю деньги каждый месяц и веду учёт расходов.")))
    assert sum(f.values()) == 0


def test_percent_only_flagged_with_unrealistic_period():
    assert flags_of(asr=asr((0, "депозит под 18% годовых")))["unrealistic_return"] == 0
    assert flags_of(asr=asr((0, "30% в неделю на автопилоте")))["unrealistic_return"] == 1
    assert flags_of(asr=asr((0, "x2 за месяц, удвоим твой депозит")))["unrealistic_return"] == 1


# ---------- категории ----------
def test_category_topic_flags():
    assert flags_of(asr=asr((0, "ставь на экспресс, кэф 2.5, промокод в описании")))["betting_terms"] == 1
    assert flags_of(asr=asr((0, "розыгрыш айфона, подпишись и поставь лайк, победитель получит приз")))["giveaway_terms"] == 1
    f = flags_of(asr=asr((0, "приведи друга и получи процент от приглашенных, твоя команда растет, уровни")))
    assert f["referral_phrase"] == 1 and f["mlm_structure_terms"] == 1
    f = flags_of(asr=asr((0, "набираю людей на удаленную работу, опыт не требуется, пиши в личку")))
    assert f["job_offer_terms"] == 1 and f["easy_money_claim"] == 1 and f["contact_call_to_action"] == 1


def test_match_cap_per_flag_source():
    segs = [(float(i), f"гарантирую доход номер {i}") for i in range(20)]
    res = extract_features(asr=asr(*segs))
    assert sum(1 for m in res["matches"] if m["flag"] == "guaranteed_income_phrase") <= 3
    assert res["flags"]["guaranteed_income_phrase"] == 1


# ---------- регрессии: баги, найденные на реальных 127 видео ----------
def test_interest_rate_and_like_requests_are_not_betting():
    for phrase in ("депозит с процентной ставкой 17% годовых", "ставьте лайки и пишите в комментариях",
                   "чик высылает тезе и ставит срок 2 дня"):
        assert flags_of(asr=asr((0, phrase)))["betting_terms"] == 0, phrase
    assert flags_of(asr=asr((0, "делай ставки на футбол, ставка на победу")))["betting_terms"] == 1


def test_link_in_bio_is_separate_weak_flag():
    f = flags_of(asr=asr((0, "переходи по ссылке в шапке профиля и забирай шаблон")))
    assert f["link_in_bio"] == 1 and f["contact_call_to_action"] == 0


def test_bare_collateral_and_passive_income_are_not_strong_claims():
    f = flags_of(asr=asr((0, "банк берет портфель как залог, это пассивный доход")))
    assert f["upfront_payment_phrase"] == 0 and f["easy_money_claim"] == 0 and f["trading_terms"] == 1


def test_earn_in_short_time_and_first_withdrawal():
    assert flags_of(asr=asr((0, "как заработать 100 долларов за 1 час")))["income_amount_claim"] == 1
    assert flags_of(asr=asr((0, "думал скам, пока не вывел первые деньги")))["income_boast"] == 1


def test_english_basics():
    assert flags_of(asr=asr((0, "add your friends into my telegram community")))["referral_phrase"] == 1
    assert flags_of(asr=asr((0, "it is risk-free and guaranteed")))["guaranteed_income_phrase"] == 1
    assert flags_of(asr=asr((0, "there is no guarantee you will profit")))["guaranteed_income_phrase"] == 0
