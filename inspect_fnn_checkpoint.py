import argparse
import torch
from pathlib import Path

from anfis_model import TemporalAttentionFNN


DEFAULT_CHECKPOINT = (
    "outputs/explicit_temporal_observation_sensitivity_6h/"
    "seed_42/observation_24h_explicit/best_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="檢查 FNN checkpoint 與臨床規則參數。")
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    return parser.parse_args()

def load_trained_model(checkpoint_path: str) -> TemporalAttentionFNN:
    """載入訓練好的模型權重"""
    print(f"正在載入模型權重: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 初始化模型 (使用與訓練時相同的設定)
    checkpoint_args = checkpoint.get('args', {})
    model_state = checkpoint['model_state_dict']
    model = TemporalAttentionFNN(
        seq_length=checkpoint_args.get('seq_length', checkpoint_args.get('input_seq_length', 24)),
        attention_hidden=checkpoint_args.get('attention_hidden', 32),
        threshold=checkpoint_args.get('threshold', 7.0),
        rule_score_scale=checkpoint_args.get('rule_score_scale', 0.2),
        use_explicit_temporal_features=model_state.get('explicit_temporal_weights') is not None,
        explicit_temporal_scale=checkpoint_args.get('explicit_temporal_scale', 1.0),
    )
    
    # 載入權重
    model.load_state_dict(model_state)
    model.eval()
    
    epoch = checkpoint.get('epoch', checkpoint.get('best_epoch', 'unknown'))
    print(f"模型載入成功，checkpoint epoch: {epoch}")
    val_metrics = checkpoint.get('val_metrics', {})
    if val_metrics:
        print(f"驗證集效能 -> AUROC: {val_metrics.get('auroc', 0):.4f} | AUPRC: {val_metrics.get('auprc', 0):.4f}")
    return model

def analyze_rule_drift(model: TemporalAttentionFNN):
    """
    評估計畫書 Section 9.4: Rule Drift
    比較「初始專家知識 (NEWS2)」與「模型訓練後」的參數差異
    """
    print("\n" + "="*50)
    print("[Rule Drift 分析] 模糊集合邊界與權重漂移")
    print("="*50)
    
    static_fnn = model.static_fnn
    
    # 我們挑選最核心的三個特徵來展示 (心跳、呼吸、血氧)
    features_to_show = ["heart_rate", "respiratory_rate", "spo2"]
    
    for feat in features_to_show:
        print(f"\n特徵: {feat.upper()}")
        print(f"{'狀態 (Linguistic Term)':<20} | {'中心點 (Center) [初始 -> 訓練後]':<35} | {'危險權重 (Weight) [初始 -> 訓練後]'}")
        print("-" * 90)
        
        # 取得初始值與訓練後的值
        init_centers = getattr(static_fnn, f"initial_centers__{feat}")
        init_weights = getattr(static_fnn, f"initial_rule_weights__{feat}")
        
        trained_centers = static_fnn.centers[feat].detach()
        trained_weights = static_fnn.rule_weights[feat].detach()
        
        term_names = static_fnn.term_names[feat]
        
        for i, term in enumerate(term_names):
            c_init = init_centers[i].item()
            c_trained = trained_centers[i].item()
            
            w_init = init_weights[i].item()
            w_trained = trained_weights[i].item()
            
            center_str = f"{c_init:>6.1f} -> {c_trained:>6.1f}"
            weight_str = f"{w_init:>5.2f} -> {w_trained:>5.2f}"
            
            print(f"{term:<20} | {center_str:<35} | {weight_str}")

def analyze_rule_complexity_and_concordance(model: TemporalAttentionFNN):
    """
    評估計畫書 Section 9.1 & 9.3: Rule Complexity & Concordance
    提取訓練後的跨特徵 IF-THEN 規則，並檢查其臨床一致性
    """
    print("\n" + "="*50)
    print("[Rule Extraction] 萃取 IF-THEN 跨特徵決策規則")
    print("="*50)
    
    static_fnn = model.static_fnn
    
    init_cross_weights = static_fnn.initial_cross_rule_weights
    trained_cross_weights = static_fnn.cross_rule_weights.detach()
    
    total_antecedents = 0
    num_rules = len(static_fnn.rule_configs)
    
    for i, rule in enumerate(static_fnn.rule_configs):
        rule_name = rule['name']
        antecedents = rule['antecedents']
        total_antecedents += len(antecedents)
        
        w_init = init_cross_weights[i].item() if len(init_cross_weights) > 0 else 0
        w_trained = trained_cross_weights[i].item()
        
        # 組合 IF-THEN 語句
        if_clause = " AND ".join([f"({feat} is {term})" for feat, term in antecedents])
        
        print(f"\nRule {i+1}: {rule_name.replace('_', ' ').title()}")
        print(f"  IF   {if_clause}")
        print(f"  THEN Deterioration Risk increases by Weight")
        print(f"  [權重變化]: 初始 = {w_init:.2f} -> 訓練後 = {w_trained:.2f}")
        
        # Concordance 檢查：如果權重被訓練成接近 0，代表模型認為這條規則無效 (Sparsity)
        # 如果權重增加，代表資料證實這條專家規則極度危險
        if w_trained < 0.1:
            print("  [Concordance] 模型將此規則靜音 (Sparsity)，可能在真實數據中較少發生或被單一特徵取代。")
        elif w_trained > w_init:
            print("  [Concordance] 資料強化了此臨床邏輯，此情境在真實 ICU 中風險較高。")
            
    # 計算 Rule Complexity (平均前件數量)
    avg_complexity = total_antecedents / max(num_rules, 1)
    print("\n" + "-"*50)
    print(f"Rule Complexity (平均每條規則的 IF 條件數): {avg_complexity:.2f}")
    print("-"*50)

if __name__ == "__main__":
    cli_args = parse_args()
    model_path = Path(cli_args.checkpoint)
    
    if Path(model_path).exists():
        trained_model = load_trained_model(model_path)
        analyze_rule_drift(trained_model)
        analyze_rule_complexity_and_concordance(trained_model)
    else:
        raise FileNotFoundError(f"找不到 checkpoint: {model_path}")
