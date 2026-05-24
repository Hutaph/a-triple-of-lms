# Spark QA Benchmark

Benchmark nay danh gia model tren dataset `data/spark_interview_questions.json`.

## Protocol

Moi record co cac truong:

- `question`: cau hoi phong van Spark/Big Data.
- `reference_answer`: dap an tham chieu ngan gon.
- `category`: nhom ky thuat.
- `difficulty`: `easy`, `medium`, hoac `hard`.
- `must_have_points`: cac y chinh bat buoc nen co trong cau tra loi.

Script se:

1. Load model chat local, mac dinh `microsoft/Phi-3-mini-4k-instruct`.
2. Hoi tung cau bang prompt co vai tro senior data engineer.
3. Luu `model_answer`, latency, so output tokens.
4. Cham tu dong bang cac metric nhe:
   - `token_f1`: muc do trung token voi dap an tham chieu.
   - `rouge_l`: muc do trung chuoi con dai nhat theo token.
   - `keyword_coverage`: ty le keyword quan trong trong reference duoc model nhac lai.
   - `must_have_point_coverage`: muc do bao phu cac y chinh bat buoc.
   - `instruction_following`: kiem tra do dai, tinh lien quan, refusal va trade-off/production wording.
   - `score`: diem tong hop = `0.25 * token_f1 + 0.20 * rouge_l + 0.25 * keyword_coverage + 0.20 * must_have_point_coverage + 0.10 * instruction_following`.
5. Tao summary theo overall, difficulty va category.
6. Tao cac chart PNG de dua vao slide/demo.
7. Tao file review thu cong va bao cao cau diem thap.

Metric nay phu hop de so sanh tuong doi giua model/prompt/config. Ket qua khong thay the human review, nhat la voi cau hoi co nhieu cach tra loi dung.

## Chay nhanh tren Colab T4

Chay 5 cau truoc de test:

```bash
python benchmarks/spark_qa_benchmark.py --limit 5 --device cuda
```

Chay full dataset:

```bash
python benchmarks/spark_qa_benchmark.py --device cuda
```

Neu chay CPU:

```bash
python benchmarks/spark_qa_benchmark.py --limit 3 --device cpu
```

## Output

Script ghi vao `benchmark_results/`:

- `spark_qa_benchmark_results.json`: chi tiet tung cau.
- `spark_qa_benchmark_results.csv`: bang ket qua de mo bang spreadsheet.
- `spark_qa_benchmark_summary.json`: summary diem trung binh.
- `human_review_template.csv`: file de cham thu cong theo rubric 1-5.
- `low_score_analysis.md`: top cau diem thap nhat kem must-have points va model answer.
- `score_by_difficulty.png`: diem trung binh theo do kho.
- `score_by_category.png`: diem trung binh theo category.
- `latency_by_difficulty.png`: latency trung binh theo do kho.
- `score_vs_latency.png`: scatter plot giua diem va latency.

## Goi y bao cao seminar

Dung cac bang sau trong slide demo:

- Average score by difficulty.
- Average score by category.
- Latency/token trung binh.
- 3 cau model tra loi tot nhat va 3 cau yeu nhat de phan tich dinh tinh.
