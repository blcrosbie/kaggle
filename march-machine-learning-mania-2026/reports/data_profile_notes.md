# Data Profile Notes

- Total CSV files profiled: **35**
- Total rows across all CSV files: **7,213,256**
- Men's files: **17**
- Women's files: **14**
- Shared files: **4**

## Largest files by rows
- `MMasseyOrdinals.csv`: 5,761,702 rows, 5 cols, missing=0.00%
- `SampleSubmissionStage1.csv`: 519,144 rows, 2 cols, missing=0.00%
- `MRegularSeasonCompactResults.csv`: 196,823 rows, 8 cols, missing=0.00%
- `WRegularSeasonCompactResults.csv`: 140,825 rows, 8 cols, missing=0.00%
- `SampleSubmissionStage2.csv`: 132,133 rows, 2 cols, missing=0.00%
- `MRegularSeasonDetailedResults.csv`: 122,775 rows, 34 cols, missing=0.00%
- `MGameCities.csv`: 90,684 rows, 6 cols, missing=0.00%
- `WGameCities.csv`: 87,353 rows, 6 cols, missing=0.00%

## Quick quality flags
- No major quality flags from missingness/duplicate checks.

## Season coverage by file
- `MConferenceTourneyGames.csv`: 2001 to 2025
- `MGameCities.csv`: 2010 to 2026
- `MMasseyOrdinals.csv`: 2003 to 2026
- `MNCAATourneyCompactResults.csv`: 1985 to 2025
- `MNCAATourneyDetailedResults.csv`: 2003 to 2025
- `MNCAATourneySeeds.csv`: 1985 to 2025
- `MNCAATourneySlots.csv`: 1985 to 2025
- `MRegularSeasonCompactResults.csv`: 1985 to 2026
- `MRegularSeasonDetailedResults.csv`: 2003 to 2026
- `MSeasons.csv`: 1985 to 2026
- `MSecondaryTourneyCompactResults.csv`: 1985 to 2025
- `MSecondaryTourneyTeams.csv`: 1985 to 2025
- `MTeamCoaches.csv`: 1985 to 2026
- `MTeamConferences.csv`: 1985 to 2026
- `WConferenceTourneyGames.csv`: 2002 to 2025
- `WGameCities.csv`: 2010 to 2026
- `WNCAATourneyCompactResults.csv`: 1998 to 2025
- `WNCAATourneyDetailedResults.csv`: 2010 to 2025
- `WNCAATourneySeeds.csv`: 1998 to 2025
- `WNCAATourneySlots.csv`: 1998 to 2025
- `WRegularSeasonCompactResults.csv`: 1998 to 2026
- `WRegularSeasonDetailedResults.csv`: 2010 to 2026
- `WSeasons.csv`: 1998 to 2026
- `WSecondaryTourneyCompactResults.csv`: 2013 to 2025
- `WSecondaryTourneyTeams.csv`: 2013 to 2025
- `WTeamConferences.csv`: 1998 to 2026

## Compact vs detailed sanity checks (preview-based)
- `MRegularSeasonCompactResults.csv` vs `MRegularSeasonDetailedResults.csv` preview minimum seasons: 1985 vs 2003 (expected detailed starts around 2003).
- `WRegularSeasonCompactResults.csv` vs `WRegularSeasonDetailedResults.csv` preview minimum seasons: 1998 vs 2010 (expected detailed starts around 2010).
- `MNCAATourneyCompactResults.csv` vs `MNCAATourneyDetailedResults.csv` preview minimum seasons: 1985 vs 2003 (expected detailed starts around 2003).
- `WNCAATourneyCompactResults.csv` vs `WNCAATourneyDetailedResults.csv` preview minimum seasons: 1998 vs 2010 (expected detailed starts around 2010).