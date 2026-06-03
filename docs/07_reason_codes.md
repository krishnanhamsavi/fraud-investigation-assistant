# Adverse Action Reason Code Mapping

## Context

Under ECOA (Equal Credit Opportunity Act) and Regulation B, lenders must provide
specific reasons when taking adverse action. For fraud scoring, the equivalent
obligation is to explain *why* a transaction was flagged — both for the fraud analyst
reviewing the case and, in a declined-transaction scenario, for the customer.

This document defines the 15 reason codes used in this system and their mapping
to the IEEE-CIS feature space.

---

## Reason Code Definitions

| Code | Description |
|------|-------------|
| R01 | Unusual transaction amount relative to account history |
| R02 | High transaction velocity or frequency on this card |
| R03 | Email domain associated with elevated fraud risk |
| R04 | Transaction inconsistent with typical geographic pattern |
| R05 | Device characteristics inconsistent with account profile |
| R06 | Transaction pattern deviates from established account behavior |
| R07 | Time-of-day or day-of-week anomaly for this account |
| R08 | Billing or shipping address information inconsistency |
| R09 | Card type or issuer profile associated with elevated risk |
| R10 | Multiple identity fields could not be verified |
| R11 | Product category associated with elevated fraud risk |
| R12 | Aggregate risk score across behavioral dimensions elevated |
| R13 | Transaction amount exceeds typical category spend |
| R14 | Cross-channel or cross-device activity pattern detected |
| R15 | Account-level risk signal from historical transaction graph |

---

## Feature-to-Code Mapping

### Direct mappings (named features)

| Feature | Reason Code | Rationale |
|---------|-------------|-----------|
| TransactionAmt, TransactionAmt_log | R01 | Amount is a primary fraud signal |
| ProductCD | R11 | Product type (W/H/C/S/R) has distinct fraud rates |
| card1–card6 | R09 | Card issuer, type, and bin-level risk |
| addr1, addr2 | R08 | Billing address mismatch signal |
| dist1, dist2 | R04 | Distance between billing/transaction locations |
| P_emaildomain, R_emaildomain | R03 | Purchaser/recipient email domain risk |
| uid_card1_email | R09 | Card×email combination pattern |
| tx_hour, tx_dayofweek | R07 | Transaction timing anomaly |
| DeviceType, DeviceInfo | R05 | Device fingerprint mismatch |

### Group mappings (regex-matched feature groups)

| Feature Group | Examples | Reason Code | Description |
|---------------|----------|-------------|-------------|
| C1–C14 | C1, C6, C13 | R02 | Vesta count features — transaction velocity |
| D1–D15 | D1, D4, D10 | R06 | Time-delta features — recency of prior activity |
| M1–M9 | M1, M4, M6 | R08 | Match flags — address/identity verification results |
| id_01–id_24 | id_01, id_05 | R10 | Numeric identity fields — SSN/DOB proxies |
| id_25–id_38 | id_30, id_31 | R05 | Device/network identity fields |
| V1–V339 | V12, V258 | R12 | Vesta proprietary features (anonymised) |

---

## Caveats

1. **V-feature anonymisation**: Vesta has not disclosed the meaning of V1–V339.
   The R12 mapping ("aggregate risk across behavioral dimensions") is conservative
   and appropriate given the opaque nature of these features.

2. **ECOA applicability**: This mapping is designed to be *directionally* consistent
   with ECOA Reg B adverse action requirements. In production, legal review of the
   exact reason code language would be required before customer-facing use.

3. **SHAP directionality**: The reason codes above apply regardless of sign. A
   transaction can be flagged *because* the amount is unusually high (R01, positive
   SHAP) or because it is suspiciously small for the card profile (R01, negative SHAP).
   The full explanation from `get_top_reasons()` includes the direction.
