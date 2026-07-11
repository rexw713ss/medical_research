# 投稿底稿實驗與數學審查

審查對象：`paper/TSP_template.tex` 與 `paper/TSP_template.pdf`  
審查版：`paper/TSP_template_review.pdf`  
更新日期：2026-07-11

## 已修正的重要問題

1. **新版與舊版結果混用**：移除舊 sequence-only FNN 主要結論，主比較改為 explicit KG-TFNN AUROC 0.6448 / AUPRC 0.1236；full-cohort 與 equal-sample 結果分開報告。
2. **重複且未完成的 Results 章節**：移除第二個重複章節、`XXXX`、空白 case-study 段落與不完整表格。
3. **模型數學式與程式不一致**：補 membership normalization、cross-rule normalization、正確 hourly risk、`G in R^(13x12)` explicit-temporal tensor，以及實作中的 output logit。
4. **Loss 寫漏**：補 class-weighted BCE、`L_cons`、`L_sparse`、`L_drift`、`L_nonneg`，並區分 optimization drift loss 與 reported membership-center drift。
5. **Training details 不完整**：補 Optuna 30 trials、8 trial epochs、AdamW、batch size、learning rate、regularization coefficients、early stopping、best epoch、GPU/software versions。
6. **缺少流程描述**：加入 8 步 training / calibration / frozen evaluation pseudocode。
7. **Outcome 與 cohort 定義不足**：明列 age >= 18、至少 4 個 SOFA components、完整 future horizon、missing-as-normal 與 complete-case sensitivity。
8. **統計方法混用**：patient-cluster bootstrap、hospital-cluster bootstrap 與 three-seed t intervals 已分開標示。
9. **主文表格過多**：14 張表重組為 7 張主表，使用 cohort、performance、ablation、sensitivity 與 rule panels。
10. **缺少投稿圖**：加入 cohort flow、architecture、ROC/PR、calibration/DCA 與 TP/FP/FN patient timelines，共 5 組圖。
11. **Discussion 結構不足**：末段改為 Contributions、Limitations、Future Work，並降低 deployment-ready 與 causal interpretation 的語氣。
12. **引用與編譯問題**：修正 bibliography fields 與 cite keys；審查版 13 筆引用均可解析。

## Feature-Matched Comparator Update

Feature-matched GRU、XGBoost 與 LightGBM 已完成。GRU 使用 24 小時 x 39 channels（13 raw、13 missingness、13 recency），tree models 使用同源 channels 的 deterministic temporal summaries；所有模型使用相同 patient split、200,000 training windows、50,000 validation windows 與完整 test windows。

Matched GRU AUROC 0.6587、LightGBM 0.6904、XGBoost 0.6870，皆高於 KG-TFNN 0.6448。底稿已移除 architecture-superiority claim，改成 predictive performance 與 intrinsic interpretability 的 tradeoff。這是結果修正，不應只放在 limitation。

## 仍需作者或外部審查

- Guideline-direction alignment 是 investigator-defined model diagnostic，不是 clinician validation；若要主張 clinical interpretability，仍需 blinded clinician adjudication 與 usability assessment。
- TRIPOD+AI checklist 需在最終排版後填入頁碼。
- PROBAST+AI 應由未參與建模的方法學者再審一次。
- 作者貢獻、IRB/waiver 措辭與資料可用性聲明需依目標期刊要求確認。
- 12/24-hour outcomes 維持 secondary analysis，不應混入 primary 6-hour claim。

## 編譯驗證

- Output：`paper/TSP_template_review.pdf`
- Pages：27
- Main tables：7
- Main figure groups：5
- Undefined citations/references：0
- Placeholder `XXXX` / `??`：0
- Remaining layout warnings：兩處 1.77 pt 的 rule-table overfull box，屬可忽略的微小排版警告。
