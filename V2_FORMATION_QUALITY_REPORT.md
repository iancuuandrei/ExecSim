# Sparse-JEPA v2 formation quality report

## Outcome

The daily-resolution formation protocol qualifies 497 of 505 candidates before ranking. Exactly 100 are frozen by median daily dollar volume with stable-ID tie-break. No target-period data, historical model, locked test, or historical TCA was run.

**AWAITING V2 FORMATION APPROVAL**

## Provider semantics

Alpaca documents that stock minute and daily bars are trade aggregates with bar-type-specific condition rules and that a bar is emitted only when all OHLCV fields are nonzero. An absent minute is therefore not proof of zero activity or zero volume. V2 uses direct SIP `1Day` bars for formation and observed-only fixed 15-minute aggregates for representation quality. It never inserts, interpolates, or zero-fills a missing minute. Sources are in `docs/RESEARCH_REFERENCES.md`.

## V1 versus v2

V1 admitted 62 names under exact full-minute formation completeness. V2 admits 497 at the unchanged 95% concept measured over 252 expected daily sessions. Of the selected 100, token completeness is high for 99, medium for 1, and low for 0 under the pre-count 95%/80% bands and 251 standard-session denominator. This is a task-resolution correction, not a threshold relaxation.

## Formation acquisition and exclusions

The direct daily corpus contains 126,461 rows for 506 observed symbols including SPY, checksum `cc0dce538534bf18f90d67bae42ad422bbd85a0c43b9642a7135826238e2fc35`. Daily exclusion reasons are `{'daily prices must be positive': 1, 'daily volume and trade count must be positive': 1, 'daily_completeness_below_95_percent': 8}`.

Complete all-candidate distributions:

```text
       daily_completeness  v1_exact_minute_completeness  token_completeness  average_observed_minute_count  median_daily_dollar_volume
count          505.000000                    505.000000          505.000000                     505.000000                5.050000e+02
mean             0.990908                      0.321802            0.977981                     357.054736                4.721254e+08
std              0.080907                      0.373137            0.093871                      51.824678                1.299684e+09
min              0.011905                      0.000000            0.011952                       4.286853                1.449952e+07
1%               0.582063                      0.000000            0.434900                     145.265020                4.408134e+07
5%               1.000000                      0.000000            0.937052                     246.402390                6.886447e+07
25%              1.000000                      0.003984            0.992032                     348.266932                1.218124e+08
50%              1.000000                      0.127490            0.992032                     379.904382                2.111631e+08
75%              1.000000                      0.665339            0.996016                     386.601594                3.840885e+08
95%              1.000000                      0.988048            1.000000                     389.800797                1.277144e+09
99%              1.000000                      1.000000            1.000000                     390.000000                5.609303e+09
max              1.000000                      1.000000            1.000000                     390.000000                2.006635e+10
```

## SPY resolution audit

On 2021-05-05 SPY has 385 observed minutes and 5 provider gaps. Full-session minute exactness is `False`, all 26 tokens are valid (`token_valid_full_session=True`), and exact TCA-window quality is `False`. The five absences therefore do not invalidate SPY for JEPA context, but they do invalidate that date for exact-minute TCA.

## Selection-bias diagnostic

For all 505 candidates, the v1 eligibility indicator has Pearson/Spearman correlations of 0.557/0.488 with log median daily dollar volume and 0.222/0.497 with average observed minute count. V1 exact-minute eligibility therefore materially favored more liquid and more continuously emitting names. The diagnostic did not change v2 eligibility or thresholds.

## Frozen top 100

| Rank | Symbol | Stable instrument | Liquidity group | Median price | Median daily dollar volume | Daily coverage | Token coverage | Band |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | TSLA | `sec-cik-0001318605-TSLA` | 1 | 729.9700 | 20,066,353,480 | 100.000% | 100.000% | high |
| 2 | AAPL | `sec-cik-0000320193-AAPL` | 1 | 141.3050 | 12,366,675,430 | 100.000% | 100.000% | high |
| 3 | AMZN | `sec-cik-0001018724-AMZN` | 1 | 3341.2300 | 11,211,441,351 | 100.000% | 100.000% | high |
| 4 | MSFT | `sec-cik-0000789019-MSFT` | 1 | 277.1650 | 7,466,748,936 | 100.000% | 100.000% | high |
| 5 | NVDA | `sec-cik-0001045810-NVDA` | 1 | 512.8800 | 5,888,989,747 | 100.000% | 100.000% | high |
| 6 | FB | `sec-cik-0001326801-FB` | 1 | 329.3650 | 5,661,037,551 | 100.000% | 100.000% | high |
| 7 | AMD | `sec-cik-0000002488-AMD` | 1 | 92.2250 | 4,367,664,832 | 100.000% | 100.000% | high |
| 8 | GOOGL | `sec-cik-0001652044-GOOGL` | 1 | 2496.2200 | 3,651,641,676 | 100.000% | 100.000% | high |
| 9 | GOOG | `sec-cik-0001652044-GOOG` | 1 | 2578.9600 | 3,093,646,023 | 100.000% | 100.000% | high |
| 10 | BA | `sec-cik-0000012927-BA` | 1 | 222.6600 | 2,539,991,859 | 100.000% | 99.203% | high |
| 11 | JPM | `sec-cik-0000019617-JPM` | 1 | 156.4900 | 2,159,692,175 | 100.000% | 99.203% | high |
| 12 | V | `sec-cik-0001403161-V` | 1 | 224.2300 | 2,076,043,168 | 100.000% | 99.203% | high |
| 13 | PYPL | `sec-cik-0001633917-PYPL` | 1 | 260.0400 | 2,024,365,240 | 100.000% | 100.000% | high |
| 14 | NFLX | `sec-cik-0001065280-NFLX` | 1 | 543.3300 | 1,999,093,493 | 100.000% | 100.000% | high |
| 15 | BAC | `sec-cik-0000070858-BAC` | 1 | 40.9000 | 1,979,038,755 | 100.000% | 99.203% | high |
| 16 | DIS | `sec-cik-0001744489-DIS` | 1 | 176.8800 | 1,727,463,695 | 100.000% | 99.203% | high |
| 17 | INTC | `sec-cik-0000050863-INTC` | 1 | 55.2250 | 1,560,165,558 | 100.000% | 100.000% | high |
| 18 | MU | `sec-cik-0000723125-MU` | 1 | 80.6600 | 1,495,622,246 | 100.000% | 100.000% | high |
| 19 | CRM | `sec-cik-0001108524-CRM` | 1 | 243.6400 | 1,458,318,343 | 100.000% | 99.203% | high |
| 20 | MA | `sec-cik-0001141391-MA` | 1 | 359.4850 | 1,445,520,710 | 100.000% | 99.203% | high |
| 21 | C | `sec-cik-0000831001-C` | 2 | 70.1300 | 1,444,638,616 | 100.000% | 99.203% | high |
| 22 | XOM | `sec-cik-0002115436-XOM` | 2 | 58.8300 | 1,429,513,819 | 100.000% | 99.203% | high |
| 23 | BRK.B | `sec-cik-0001067983-BRK-B` | 2 | 278.7900 | 1,361,040,214 | 100.000% | 99.203% | high |
| 24 | JNJ | `sec-cik-0000200406-JNJ` | 2 | 164.6700 | 1,307,036,178 | 100.000% | 99.203% | high |
| 25 | PFE | `sec-cik-0000078003-PFE` | 2 | 39.9950 | 1,286,452,447 | 100.000% | 99.203% | high |
| 26 | UNH | `sec-cik-0000731766-UNH` | 2 | 408.5950 | 1,281,599,813 | 100.000% | 99.203% | high |
| 27 | ADBE | `sec-cik-0000796343-ADBE` | 2 | 569.3250 | 1,259,320,186 | 100.000% | 100.000% | high |
| 28 | HD | `sec-cik-0000354950-HD` | 2 | 323.6350 | 1,247,395,207 | 100.000% | 99.203% | high |
| 29 | WFC | `sec-cik-0000072971-WFC` | 2 | 45.7500 | 1,241,021,278 | 100.000% | 99.203% | high |
| 30 | QCOM | `sec-cik-0000804328-QCOM` | 2 | 140.3350 | 1,225,655,130 | 100.000% | 100.000% | high |
| 31 | CVX | `sec-cik-0000093410-CVX` | 2 | 103.9900 | 1,165,776,100 | 100.000% | 99.203% | high |
| 32 | WMT | `sec-cik-0000104169-WMT` | 2 | 141.5800 | 1,157,914,725 | 100.000% | 99.203% | high |
| 33 | T | `sec-cik-0000732717-T` | 2 | 28.4650 | 1,153,206,059 | 100.000% | 99.203% | high |
| 34 | PG | `sec-cik-0000080424-PG` | 2 | 138.0000 | 1,131,111,207 | 100.000% | 99.203% | high |
| 35 | VZ | `sec-cik-0000732712-VZ` | 2 | 55.6600 | 1,075,461,461 | 100.000% | 99.203% | high |
| 36 | AMAT | `sec-cik-0000006951-AMAT` | 2 | 134.6250 | 1,066,295,137 | 100.000% | 100.000% | high |
| 37 | CSCO | `sec-cik-0000858877-CSCO` | 2 | 53.6050 | 1,045,567,540 | 100.000% | 100.000% | high |
| 38 | GS | `sec-cik-0000886982-GS` | 2 | 373.9950 | 1,036,423,203 | 100.000% | 99.203% | high |
| 39 | GM | `sec-cik-0001467858-GM` | 2 | 56.5950 | 1,009,776,238 | 100.000% | 99.203% | high |
| 40 | F | `sec-cik-0000037996-F` | 2 | 13.4300 | 990,042,501 | 100.000% | 99.203% | high |
| 41 | NKE | `sec-cik-0000320187-NKE` | 3 | 148.4200 | 974,572,441 | 100.000% | 99.203% | high |
| 42 | CMCSA | `sec-cik-0001166691-CMCSA` | 3 | 55.0850 | 974,043,770 | 100.000% | 100.000% | high |
| 43 | MRK | `sec-cik-0000310158-MRK` | 3 | 77.0250 | 971,518,532 | 100.000% | 99.203% | high |
| 44 | MS | `sec-cik-0000895421-MS` | 3 | 92.2150 | 933,957,766 | 100.000% | 99.203% | high |
| 45 | AVGO | `sec-cik-0001730168-AVGO` | 3 | 480.3450 | 928,000,546 | 100.000% | 100.000% | high |
| 46 | TWTR | `sec-cik-0001418091-TWTR` | 3 | 62.1050 | 926,064,198 | 100.000% | 98.805% | high |
| 47 | COST | `sec-cik-0000909832-COST` | 3 | 398.9000 | 922,929,540 | 100.000% | 100.000% | high |
| 48 | ORCL | `sec-cik-0001341439-ORCL` | 3 | 84.5650 | 907,690,815 | 100.000% | 99.203% | high |
| 49 | KO | `sec-cik-0000021344-KO` | 3 | 54.4400 | 886,027,538 | 100.000% | 99.602% | high |
| 50 | TXN | `sec-cik-0000097476-TXN` | 3 | 188.4150 | 874,893,721 | 100.000% | 100.000% | high |
| 51 | LRCX | `sec-cik-0000707549-LRCX` | 3 | 608.2550 | 874,576,111 | 100.000% | 100.000% | high |
| 52 | GE | `sec-cik-0000040545-GE` | 3 | 13.5700 | 852,255,005 | 100.000% | 99.203% | high |
| 53 | TMO | `sec-cik-0000097745-TMO` | 3 | 514.6450 | 822,306,628 | 100.000% | 99.203% | high |
| 54 | ABBV | `sec-cik-0001551152-ABBV` | 3 | 112.3300 | 803,992,382 | 100.000% | 99.602% | high |
| 55 | BKNG | `sec-cik-0001075531-BKNG` | 3 | 2301.2100 | 786,995,988 | 100.000% | 92.430% | medium |
| 56 | CCL | `sec-cik-0000815097-CCL` | 3 | 23.6400 | 775,812,802 | 100.000% | 99.203% | high |
| 57 | PEP | `sec-cik-0000077476-PEP` | 3 | 149.0200 | 764,078,334 | 100.000% | 100.000% | high |
| 58 | FCX | `sec-cik-0000831259-FCX` | 3 | 36.5150 | 757,425,731 | 100.000% | 99.203% | high |
| 59 | BMY | `sec-cik-0000014272-BMY` | 3 | 62.7450 | 750,204,816 | 100.000% | 99.203% | high |
| 60 | LOW | `sec-cik-0000060667-LOW` | 3 | 195.7850 | 748,676,094 | 100.000% | 99.203% | high |
| 61 | TGT | `sec-cik-0000027419-TGT` | 4 | 231.2400 | 746,938,587 | 100.000% | 99.203% | high |
| 62 | NOW | `sec-cik-0001373715-NOW` | 4 | 570.4650 | 723,727,074 | 100.000% | 99.203% | high |
| 63 | CAT | `sec-cik-0000018230-CAT` | 4 | 209.1500 | 723,111,578 | 100.000% | 99.203% | high |
| 64 | LLY | `sec-cik-0000059478-LLY` | 4 | 229.5550 | 719,202,788 | 100.000% | 99.203% | high |
| 65 | DHR | `sec-cik-0000313616-DHR` | 4 | 275.0750 | 706,958,553 | 100.000% | 99.203% | high |
| 66 | IBM | `sec-cik-0000051143-IBM` | 4 | 136.0750 | 703,266,393 | 100.000% | 99.203% | high |
| 67 | SBUX | `sec-cik-0000829224-SBUX` | 4 | 112.4850 | 695,651,036 | 100.000% | 100.000% | high |
| 68 | MCD | `sec-cik-0000063908-MCD` | 4 | 235.0100 | 695,457,831 | 100.000% | 99.203% | high |
| 69 | AAL | `sec-cik-0000006201-AAL` | 4 | 20.4450 | 685,874,908 | 100.000% | 100.000% | high |
| 70 | CHTR | `sec-cik-0001091667-CHTR` | 4 | 682.5500 | 684,257,478 | 100.000% | 100.000% | high |
| 71 | ABT | `sec-cik-0000001800-ABT` | 4 | 120.8500 | 683,122,855 | 100.000% | 99.602% | high |
| 72 | UNP | `sec-cik-0000100885-UNP` | 4 | 221.0400 | 673,076,215 | 100.000% | 99.203% | high |
| 73 | AMGN | `sec-cik-0000318154-AMGN` | 4 | 233.7850 | 662,739,614 | 100.000% | 100.000% | high |
| 74 | ACN | `sec-cik-0001467373-ACN` | 4 | 305.1200 | 657,440,219 | 100.000% | 99.602% | high |
| 75 | INTU | `sec-cik-0000896878-INTU` | 4 | 498.2100 | 646,353,864 | 100.000% | 100.000% | high |
| 76 | NEE | `sec-cik-0000753308-NEE` | 4 | 79.7700 | 631,436,031 | 100.000% | 99.203% | high |
| 77 | TMUS | `sec-cik-0001283699-TMUS` | 4 | 129.6400 | 625,493,341 | 100.000% | 100.000% | high |
| 78 | UAL | `sec-cik-0000100517-UAL` | 4 | 48.3500 | 621,303,474 | 100.000% | 100.000% | high |
| 79 | ATVI | `sec-cik-0000718877-ATVI` | 4 | 90.4800 | 619,018,785 | 100.000% | 100.000% | high |
| 80 | FDX | `sec-cik-0001048911-FDX` | 4 | 261.9750 | 604,712,169 | 100.000% | 99.203% | high |
| 81 | MDT | `sec-cik-0001613103-MDT` | 5 | 123.1750 | 602,391,782 | 100.000% | 99.203% | high |
| 82 | DE | `sec-cik-0000315189-DE` | 5 | 352.7150 | 601,864,562 | 100.000% | 99.203% | high |
| 83 | VIAC | `sec-cik-0000813828-VIAC` | 5 | 40.7500 | 601,469,142 | 100.000% | 100.000% | high |
| 84 | AXP | `sec-cik-0000004962-AXP` | 5 | 161.7500 | 596,623,647 | 100.000% | 99.203% | high |
| 85 | HON | `sec-cik-0000773840-HON` | 5 | 218.9350 | 592,352,372 | 100.000% | 99.602% | high |
| 86 | SPGI | `sec-cik-0000064040-SPGI` | 5 | 411.3500 | 590,924,382 | 100.000% | 99.203% | high |
| 87 | UPS | `sec-cik-0001090727-UPS` | 5 | 194.0950 | 571,333,483 | 100.000% | 99.203% | high |
| 88 | ADI | `sec-cik-0000006281-ADI` | 5 | 164.3050 | 570,196,704 | 100.000% | 100.000% | high |
| 89 | LMT | `sec-cik-0000936468-LMT` | 5 | 356.0600 | 554,237,012 | 100.000% | 99.203% | high |
| 90 | COP | `sec-cik-0001163165-COP` | 5 | 56.8500 | 551,827,756 | 100.000% | 99.203% | high |
| 91 | CVS | `sec-cik-0000064803-CVS` | 5 | 83.7900 | 520,428,734 | 100.000% | 99.203% | high |
| 92 | DAL | `sec-cik-0000027904-DAL` | 5 | 42.3000 | 514,347,964 | 100.000% | 99.203% | high |
| 93 | ETSY | `sec-cik-0001370637-ETSY` | 5 | 209.5200 | 513,936,260 | 100.000% | 100.000% | high |
| 94 | SCHW | `sec-cik-0000316709-SCHW` | 5 | 71.6700 | 511,671,134 | 100.000% | 99.602% | high |
| 95 | GILD | `sec-cik-0000882095-GILD` | 5 | 67.4800 | 507,743,684 | 100.000% | 100.000% | high |
| 96 | ISRG | `sec-cik-0001035267-ISRG` | 5 | 802.3600 | 507,478,823 | 100.000% | 99.602% | high |
| 97 | AMT | `sec-cik-0001053507-AMT` | 5 | 265.7100 | 507,007,618 | 100.000% | 99.203% | high |
| 98 | RTX | `sec-cik-0000101829-RTX` | 5 | 84.8350 | 506,285,842 | 100.000% | 99.203% | high |
| 99 | LIN | `sec-cik-0001707925-LIN` | 5 | 296.2950 | 505,675,370 | 100.000% | 99.203% | high |
| 100 | FIS | `sec-cik-0001136893-FIS` | 5 | 133.9750 | 500,032,508 | 100.000% | 99.203% | high |

## Resource evidence

The bounded scan read 45,464,276 minute rows from 1.439 GiB of response files in 1516.73 seconds: 29,975.13 minute rows/s and 2,167.76 token attempts/s. The quality Parquet is 2.78 MiB. Peak RSS: >= 255.16 MiB (OS lifetime-peak poll; in-process terminal counter unavailable in this run).

## Target acquisition plan

The frozen 100 plus SPY imply 4,850 monthly symbol-interval requests and at most 39,508,170 minute rows over 1,003 target XNYS sessions. Projecting only for capacity planning from formation token availability gives approximately 99,830 stock/SPY token sessions. The embedding upper bound is 74.83 GiB. These are planning quantities, not acquired data or empirical results.

## Evidence boundary

- DATA: formation only
- TARGET DATA: NOT RUN
- HISTORICAL TRAINING: NOT RUN
- LOCKED TEST: NOT RUN
- TCA: NOT RUN

**AWAITING V2 FORMATION APPROVAL**
