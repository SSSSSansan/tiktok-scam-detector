"""
Чистит сырую таблицу (Book1Sheet1.csv) -> data/labels.csv

Запуск из корня репозитория:
    python tools/clean_labels.py data/raw_labels.csv data/labels.csv
"""
import csv
import re
import sys
from collections import Counter

CATEGORIES = ["investment_platform", "pyramid", "betting", "giveaway", "job_offer", "none"]
SCAM_CATEGORIES = [c for c in CATEGORIES if c != "none"]

CATEGORY_ALIASES = {
    "investment platform": "investment_platform",
    "job offer": "job_offer",
}

TT_FULL = re.compile(r"tiktok\.com/@([^/?#]+)/(video|photo)/(\d+)")
TT_SHORT = re.compile(r"(?:vt|vm)\.tiktok\.com/([A-Za-z0-9]+)")
IG_REEL = re.compile(r"instagram\.com/(?:reel|reels)/([A-Za-z0-9_-]+)")

OUT_COLUMNS = ["video_id", "url", "platform", "category", "risk_human", "content_type", "notes"]


def norm_category(raw):
    c = raw.strip().lower()
    return CATEGORY_ALIASES.get(c, c.replace(" ", "_"))


def parse_url(raw):
    """Возвращает (video_id, чистая_ссылка, платформа, тип) или None."""
    url = raw.strip().replace("\n", "").replace("\r", "")
    m = TT_FULL.search(url)
    if m:
        user, kind, vid = m.groups()
        return f"tt_{vid}", f"https://www.tiktok.com/@{user}/{kind}/{vid}", "tiktok", kind
    m = TT_SHORT.search(url)
    if m:
        code = m.group(1)
        return f"tt_short_{code}", f"https://vt.tiktok.com/{code}", "tiktok", "unknown"
    m = IG_REEL.search(url)
    if m:
        code = m.group(1)
        return f"ig_{code}", f"https://www.instagram.com/reel/{code}/", "instagram", "video"
    return None


def main(src, dst):
    with open(src, encoding="utf-8-sig", newline="") as f:
        records = list(csv.reader(f, delimiter=";"))

    out, seen, problems = [], {}, []

    # records[0] - заголовок, значит запись i соответствует строке i+1 в Excel/Google Sheets
    for i, row in enumerate(records[1:], start=2):
        row = [c.strip() for c in row]
        if not any(row):
            continue
        if len(row) < 3:
            problems.append(f"строка {i}: меньше 3 колонок -> {row}")
            continue

        raw_url, raw_risk, raw_cat = row[0], row[1], row[2]

        parsed = parse_url(raw_url)
        if parsed is None:
            problems.append(f"строка {i}: нет ссылки в первой колонке -> {raw_url[:60]!r} (пропущена)")
            continue
        video_id, url, platform, content_type = parsed

        category = norm_category(raw_cat)
        if category not in CATEGORIES:
            problems.append(f"строка {i}: неизвестная категория {raw_cat!r} (пропущена)")
            continue

        try:
            risk = float(raw_risk.replace(",", "."))
            if not 0.0 <= risk <= 1.0:
                raise ValueError
        except ValueError:
            problems.append(f"строка {i}: плохой risk_score {raw_risk!r} (пропущена)")
            continue

        if len(row) > 3 and row[3] and row[3].lower() != platform:
            problems.append(f"строка {i}: платформа в колонке {row[3]!r}, а по ссылке {platform!r} (взяла по ссылке)")

        if video_id in seen:
            prev = seen[video_id]
            if prev["category"] == category:
                problems.append(f"строка {i}: дубль {video_id} (пропущена)")
            else:
                problems.append(
                    f"строка {i}: ТОТ ЖЕ {video_id} с другой категорией: "
                    f"{prev['category']} и {category} (оставила первую)"
                )
            continue

        notes = []
        if content_type == "photo":
            notes.append("photo_post")
        if category == "none" and risk >= 0.5:
            notes.append("check_risk")
        if category != "none" and risk <= 0.2:
            notes.append("check_risk")
        if notes and "check_risk" in notes:
            problems.append(f"строка {i}: проверь: категория {category}, но risk={risk}")

        rec = {
            "video_id": video_id,
            "url": url,
            "platform": platform,
            "category": category,
            "risk_human": risk,
            "content_type": content_type,
            "notes": ",".join(notes),
        }
        seen[video_id] = rec
        out.append(rec)

    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(out)

    # ---- отчёт ----
    print(f"\nГотово: {len(out)} видео записано в {dst}\n")
    print("Категория x платформа:")
    print(f"  {'category':<22}{'tiktok':>8}{'instagram':>11}{'всего':>8}")
    counts = Counter((r["category"], r["platform"]) for r in out)
    for c in CATEGORIES:
        tt, ig = counts[(c, "tiktok")], counts[(c, "instagram")]
        print(f"  {c:<22}{tt:>8}{ig:>11}{tt + ig:>8}")
    print(f"  {'ИТОГО':<22}{sum(v for (c, p), v in counts.items() if p == 'tiktok'):>8}"
          f"{sum(v for (c, p), v in counts.items() if p == 'instagram'):>11}{len(out):>8}")
    shorts = sum(1 for r in out if r["video_id"].startswith("tt_short_"))
    photos = sum(1 for r in out if r["content_type"] == "photo")
    print(f"\nКоротких ссылок vt.tiktok.com: {shorts} | фото-постов (без видео): {photos}")

    if problems:
        print(f"\nЧто проверить вручную ({len(problems)}):")
        for p in problems:
            print("  -", p)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Использование: python tools/clean_labels.py data/raw_labels.csv data/labels.csv")
    main(sys.argv[1], sys.argv[2])