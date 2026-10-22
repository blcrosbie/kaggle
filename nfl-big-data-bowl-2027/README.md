# NFL Big Data Bowl 2027

## Setup
```bash
cd nfl-big-data-bowl-2027
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1. kaggle.com -> Settings -> API -> Create New Token  => ~/.kaggle/kaggle.json (chmod 600)
# 2. Join the competition + accept rules in the browser
./scripts/download_data.sh
```

## Plan
1. **Profile** data: schema, ID linkage Combine -> NFL, drills available, n per position/year.
2. **Signal processing**: clean 10 Hz traces; derive per-rep features (accel/decel profile, jerk, change-of-direction
   angle & speed loss, stride-frequency spectrum, time-to-peak-speed, rep-to-rep variability); unsupervised anomaly
   detection (e.g. matrix profile / isolation forest on rep features) to flag "hidden" movers whose 40/3-cone times look average.
3. **Link to NFL**: pick ONE position group + ONE outcome with enough snaps (e.g. DB separation/coverage,
   WR separation off breaks, DL pressure). Hierarchical/shrinkage models; leave-one-draft-class-out CV.
4. **Betting-market layer (companion, not core)**: map signals to publicly posted player props / futures; see notes.
5. Writeup + public notebook + viral explainer.
