# Adversarial report

| Кейс | Тип | Ожидали | risk_pred | Категория | Итог |
|---|---|---|---|---|---|
| A01 | no_speech | betting (high) | 0.70 | betting | OK |
| A02 | no_speech | betting (high) | 0.70 | betting | OK |
| A03 | no_text | pyramid (high) | 0.60 | pyramid | OK |
| A04 | no_text | pyramid (high) | 0.60 | pyramid | OK |
| A05 | kazakh | none (low) | 0.20 | none | OK |
| A06 | kazakh | investment_platform (high) | 0.60 | investment_platform | OK |
| A07 | obfuscation | betting (high) | 0.70 | betting | OK |
| A08 | obfuscation | betting (high) | 0.70 | betting | OK |
| A09 | hard_negative | none (low) | 0.20 | none | OK |
| A10 | hard_negative | none (low) | 0.20 | none | OK |
| A11 | telegram_scam | pyramid (high) | 0.20 | none | FAIL |
| A12 | telegram_scam | job_offer (high) | 0.60 | job_offer | OK |

Пройдено: 11/12. Порог 0.5 для оценки low/high выбран нами и не подбирался под данные.