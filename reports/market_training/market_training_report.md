# Market-wide LSTM candidate training

Data cutoff (exclusive): 2026-10-11

Validation: eligible=34, ineligible=16, invalid=39

Raw data is unchanged. No missing prices/sessions are filled. Invalid OHLC blocks a symbol. Stale/short histories are retained for research but excluded from current candidate training. Instrument type/listing status is not verified by this CSV validation.

Training uses chronological 60/20/20 splits with horizon purging; scalers fit only training inputs. The final holdout is not scored or fitted. These are development candidates, not deployment approvals.

| Symbol | Data status | Rows | Latest date | Training | Validation RMSE | Baseline RMSE | Reason |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| AB | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| ADWYA | ineligible | 3145 | 2022-12-30 | not trained | — | — | Stale last observation (1381 days) |
| AETEC | ineligible | 1847 | 2026-07-22 | not trained | — | — | Stale last observation (81 days) |
| AL | eligible | 2753 | 2026-10-08 | candidate | 0.01927632954333862 | 0.019363985520415395 |  |
| ALKIM | eligible | 1194 | 2026-09-28 | candidate | 0.03453081736751882 | 0.035943371864732614 |  |
| AMI | invalid | — | — | not trained | — | — | [Errno 2] No such file or directory: 'data\\raw\\AMI.csv' |
| AMS | ineligible | 2825 | 2023-07-05 | not trained | — | — | Stale last observation (1194 days) |
| AMV | eligible | 874 | 2026-10-09 | candidate | 0.012300148200862952 | 0.012349913205069054 |  |
| ARTES | eligible | 4075 | 2026-10-07 | rejected | 0.015269707551184446 | 0.015257800795798184 |  |
| ASSAD | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| ASSMA | eligible | 886 | 2026-10-07 | candidate | 0.014502612669682259 | 0.015675177608017902 |  |
| AST | eligible | 959 | 2026-10-07 | rejected | 0.01808363171623343 | 0.017892304405574305 |  |
| ATB | eligible | 3986 | 2026-10-07 | candidate | 0.016148144435912545 | 0.016182714537730777 |  |
| ATL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| BH | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| BHASS | eligible | 1608 | 2026-09-28 | candidate | 0.028010874543356174 | 0.02804484814447925 |  |
| BHL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| BIAT | eligible | 4127 | 2026-10-07 | candidate | 0.012440769543346298 | 0.012449514831574053 |  |
| BL | eligible | 1175 | 2026-10-07 | candidate | 0.04066799703845156 | 0.041046972115171984 |  |
| BNA | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| BT | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| BTE | eligible | 2726 | 2026-10-07 | candidate | 0.026653641613165538 | 0.02714357519609879 |  |
| CC | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| CELL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| CIL | eligible | 3009 | 2026-10-07 | candidate | 0.025266010100928327 | 0.025486208200803592 |  |
| CITY | eligible | 3022 | 2026-10-07 | rejected | 0.015011403457212061 | 0.014989135289904944 |  |
| CREAL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| DH | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| ECYCL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| ELBEN | ineligible | 803 | 2019-10-14 | not trained | — | — | Stale last observation (2554 days) |
| GIF | ineligible | 3232 | 2024-10-22 | not trained | — | — | Stale last observation (719 days) |
| HL | eligible | 1722 | 2026-10-05 | candidate | 0.02606097463674645 | 0.026451318207671912 |  |
| ICF | eligible | 3033 | 2026-10-09 | rejected | 0.015515692833890189 | 0.015482984884075457 |  |
| LNDOR | eligible | 3096 | 2026-10-09 | rejected | 0.017335498594369832 | 0.017286097189079336 |  |
| LSTR | ineligible | 2892 | 2024-07-22 | not trained | — | — | Stale last observation (811 days) |
| MAG | eligible | 2658 | 2026-10-08 | candidate | 0.021478345576017216 | 0.02166987713407183 |  |
| MGR | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| MIP | ineligible | 588 | 2024-07-29 | not trained | — | — | Stale last observation (804 days) |
| MNP | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| MPBS | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| NAKL | eligible | 3761 | 2026-10-07 | rejected | 0.013876710098959652 | 0.013812834984539184 |  |
| NBL | eligible | 2964 | 2026-10-09 | rejected | 0.01760650437887406 | 0.017582233195042072 |  |
| OTH | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| PGH | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| PLAST | eligible | 2004 | 2026-10-09 | candidate | 0.03667913866214295 | 0.036856629312359446 |  |
| PLTU | ineligible | 353 | 2026-10-01 | not trained | — | — | Insufficient samples for train/validation/holdout minimums |
| PX1 | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SAH | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SAM | eligible | 2299 | 2026-10-09 | candidate | 0.01634862148404285 | 0.01639410828747649 |  |
| SCB | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SERVI | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SFBT | eligible | 4167 | 2026-10-09 | candidate | 0.013221342217905514 | 0.013228922646847347 |  |
| SIAME | eligible | 3962 | 2026-10-08 | candidate | 0.01503774281699819 | 0.01514514226119238 |  |
| SIMPA | ineligible | 2174 | 2026-07-31 | not trained | — | — | Stale last observation (72 days) |
| SIPHA | eligible | 1939 | 2026-10-05 | candidate | 0.04580347003742472 | 0.04637021000263921 |  |
| SITS | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SMART | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SMD | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SOKNA | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SOMOC | eligible | 4062 | 2026-10-09 | candidate | 0.017002727065436662 | 0.017181233996371177 |  |
| SOPAT | ineligible | 3092 | 2023-10-31 | not trained | — | — | Stale last observation (1076 days) |
| SOTEM | ineligible | 1164 | 2026-09-24 | not trained | — | — | Stale last observation (17 days) |
| SOTET | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SOTUV | eligible | 4138 | 2026-10-09 | rejected | 0.015737011956248305 | 0.015695464165321207 |  |
| SPDIT | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| SPHAX | ineligible | 345 | 2014-11-17 | not trained | — | — | Stale last observation (4346 days) |
| STA | eligible | 1014 | 2026-10-09 | candidate | 0.019904960704146043 | 0.020125032316064138 |  |
| STAR | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| STB | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| STEQ | ineligible | 565 | 2020-12-30 | not trained | — | — | Stale last observation (2111 days) |
| STIP | eligible | 1424 | 2026-10-05 | candidate | 0.047222463369640914 | 0.04753525018216415 |  |
| STPAP | eligible | 3043 | 2026-10-09 | candidate | 0.015402864399285864 | 0.015449813882422262 |  |
| STPIL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TAIR | eligible | 3945 | 2026-10-08 | candidate | 0.018971769349133905 | 0.01912766865303366 |  |
| TBIDX | ineligible | 3599 | 2026-09-25 | not trained | — | — | Stale last observation (16 days) |
| TGH | eligible | 2797 | 2026-10-09 | candidate | 0.024650388704565644 | 0.024820966994779807 |  |
| TINV | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TJARI | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TJL | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TLNET | eligible | 3802 | 2026-10-09 | rejected | 0.01226113472278617 | 0.0122397906291247 |  |
| TLS | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TPR | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TRE | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| TVAL | ineligible | 391 | 2020-07-21 | not trained | — | — | Stale last observation (2273 days) |
| UADH | ineligible | 2122 | 2026-07-22 | not trained | — | — | Stale last observation (81 days) |
| UBCI | eligible | 2132 | 2026-10-09 | candidate | 0.0229656857927404 | 0.023009368143377215 |  |
| UIB | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| UMED | invalid | — | — | not trained | — | — | OHLC range inconsistency |
| WIFAK | eligible | 2884 | 2026-10-09 | candidate | 0.0172394670256565 | 0.017409198758455903 |  |

Full dataset lineage: `data/processed/manifest.json` (DVC tracked). Training records: `artifacts/market/training/`; model weights: `artifacts/models/`; MLflow contains each training run and registered candidate version.
