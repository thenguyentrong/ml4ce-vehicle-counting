# Crossing audit checklist — fine-tuned + Hungarian

47 crossings the system claims, in time order. Scrub to each timestamp in
`runs/part2/manual/reference.mp4` and tick it off.

**Also do one independent pass** watching straight through without this list — it can only
catch false positives, never vehicles the system MISSED, and misses are the larger error.

| # | frame | time | direction claimed | real vehicle? |
|---|---|---|---|---|
| 1 | 2 | 0:00.07 | away from camera | |
| 2 | 28 | 0:00.93 | toward camera | |
| 3 | 35 | 0:01.17 | away from camera | |
| 4 | 62 | 0:02.07 | toward camera | |
| 5 | 65 | 0:02.17 | away from camera | |
| 6 | 93 | 0:03.10 | away from camera | |
| 7 | 124 | 0:04.14 | away from camera | |
| 8 | 128 | 0:04.27 | toward camera | |
| 9 | 140 | 0:04.67 | toward camera | |
| 10 | 144 | 0:04.80 | toward camera | |
| 11 | 226 | 0:07.54 | toward camera | |
| 12 | 240 | 0:08.01 | toward camera | |
| 13 | 303 | 0:10.11 | toward camera | |
| 14 | 376 | 0:12.55 | toward camera | |
| 15 | 451 | 0:15.05 | toward camera | |
| 16 | 462 | 0:15.42 | away from camera | |
| 17 | 470 | 0:15.68 | toward camera | |
| 18 | 496 | 0:16.55 | toward camera | |
| 19 | 512 | 0:17.08 | toward camera | |
| 20 | 552 | 0:18.42 | toward camera | |
| 21 | 561 | 0:18.72 | toward camera | |
| 22 | 616 | 0:20.55 | away from camera | |
| 23 | 618 | 0:20.62 | toward camera | |
| 24 | 622 | 0:20.75 | away from camera | |
| 25 | 622 | 0:20.75 | away from camera | |
| 26 | 627 | 0:20.92 | toward camera | |
| 27 | 629 | 0:20.99 | toward camera | |
| 28 | 671 | 0:22.39 | away from camera | |
| 29 | 686 | 0:22.89 | toward camera | |
| 30 | 695 | 0:23.19 | toward camera | |
| 31 | 701 | 0:23.39 | toward camera | |
| 32 | 704 | 0:23.49 | toward camera | |
| 33 | 745 | 0:24.86 | toward camera | |
| 34 | 757 | 0:25.26 | toward camera | |
| 35 | 763 | 0:25.46 | toward camera | |
| 36 | 802 | 0:26.76 | away from camera | |
| 37 | 816 | 0:27.23 | toward camera | |
| 38 | 825 | 0:27.53 | toward camera | |
| 39 | 861 | 0:28.73 | away from camera | |
| 40 | 894 | 0:29.83 | toward camera | |
| 41 | 894 | 0:29.83 | away from camera | |
| 42 | 1000 | 0:33.37 | away from camera | |
| 43 | 1031 | 0:34.40 | away from camera | |
| 44 | 1035 | 0:34.53 | toward camera | |
| 45 | 1127 | 0:37.60 | toward camera | |
| 46 | 1198 | 0:39.97 | toward camera | |
| 47 | 1247 | 0:41.61 | toward camera | |

## Tally

| | toward camera | away from camera | total |
|---|---|---|---|
| system claims | 32 | 15 | 47 |
| of those, real | | | |
| vehicles system MISSED | | | |
| **manual truth** | **23** | **20** | **43** |
| **error** | **+9** | **−5** | **+4** |

The manual truth row is recorded (`docs/manual_count.md`, second pass 23.08). The two blank rows
still need this checklist worked through: they decompose the +9 into double counts versus phantom
tracks, and the −5 into vehicles the detector never saw versus tracks that died before the line.
