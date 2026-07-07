# scripts/lgbtq/

Drop the **lgbtq** trait's original scripts here (from your `lgbtq/` project folder):

- 00_finetune_teacher.py
- 01_verify_and_baseline.py
- 02_generate_tweets.py
- 03_semantic_filter.py
- 04_finetune_student.py
- 04_finetune_students_crossmodels.py
- 05_evaluate_student.py
- 05_evaluate_student_crossmodels.py
- Teacher_seed_dataset.json
- topics.txt
- tweet_stats.json

These are kept SEPARATE per trait on purpose (Option A). Do not merge them with
other traits' scripts — trait-specific logic may differ even where filenames match.
Delete this PLACEHOLDER.md once the real files are in.
