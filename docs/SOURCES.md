# Source and license register

| Series | Official/provider page | Local data path | Use and redistribution |
| --- | --- | --- | --- |
| BTC/USD | Bundled CSV; original URL unknown | `data/crypto_statistics_data.csv` | Upstream license unknown. Do not claim redistribution rights. |
| SPY adjusted daily | [Yahoo Finance SPY history](https://finance.yahoo.com/quote/SPY/history/) | ignored `data/raw/spy_chart.json` | Research use locally. Raw-feed redistribution rights unverified; no raw response committed. |
| EUR/USD reference | [ECB series](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A) | ignored `data/raw/ecb_eurusd.csv` | [ECB policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html) allows reuse with attribution and disclosure of modifications. |
| WTI Cushing spot | [EIA daily series](https://www.eia.gov/dnav/pet/hist/RWTCD.htm) | ignored `data/raw/eia_wti.xls` | [EIA policy](https://www.eia.gov/about/copyrights_reuse.php) permits reuse of government data with attribution. |

The source URL actually requested, raw SHA-256, local cache modification time, symbol mapping, timezone, currency, adjustment, calendar and target meaning are saved per run in `data_manifest.json`. SPY uses Yahoo's adjusted close as the target but retains raw OHLC. ECB is a reference fixing and EIA is a spot assessment; both are point prices with null OHLC. The WTI series is not continuous futures and therefore has no roll construction or roll jumps. Market calendars are preserved by retaining only observed source dates.
