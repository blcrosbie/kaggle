# Competition Brief

## Core constraints
- maximum of five (5) Submissions per day. b. You may select up to two (2) Final Submissions for judging.
- select up to two (2) Final Submissions for judging.
- maximum Team size is five (5). b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.
- You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations).

## Evaluation and submission format
- Submissions are evaluated on the Brier score between the predicted probabilities and the actual game outcomes (this is equivalent to mean squared error in this context).
- Single submission must include both men's and women's matchups.
- You must predict the probability that the team with the lower TeamId beats the team with the higher TeamId. Note that the men's teams and women's TeamIds do not overlap.
- Required output columns: `ID,Pred` with `ID` like `2026_1101_1102`.
- leaderboard of this competition will be only meaningful once the 2026 tournaments begin and Kaggle rescores your predictions!

## Timeline (from overview.md)
- February 19, 2026 - Start Date
- March 19, 2026 4PM UTC - Final Submission Deadline. Note that Kaggle will release updated data at least once in advance of the deadline in order to include as much of the current season's data as possible.

## Data inventory highlights
- Data sections documented: 25
- Files are split by prefixes: `M*` men, `W*` women, shared files without prefix.
- Historical season depth reaches back to 1985 in core men's files and 1998 in core women's files.
- Detailed box-score files begin later (men 2003, women 2010).

## Notes
- `about/overview.html` is primarily Kaggle page shell markup (scripts/meta), with no additional competition-rule content beyond metadata/title.
- Keep external data/tools reasonably accessible and low-cost per rules.

## Immediate execution plan
1. Profile all CSVs and confirm season coverage, missingness, and key joins.
2. Build a baseline model from regular season + tournament history.
3. Generate 5 daily candidate submissions and track outcomes.
4. Promote top daily candidates to the 2 final submissions near deadline.