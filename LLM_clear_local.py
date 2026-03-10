import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import csv

# ===== 配置 =====
print("🔄 加载模型...")
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",  # 正确ID，顶级模型
    device=device,
    batch_size=16  # 更快
)

print("✅ 模型加载完成！\n")

candidate_labels = [
    "related to USDT, Tether, or stablecoins including discussions about reserves, transparency, depeg, company operations, audits, regulation, or comparison with other stablecoins",
    "unrelated to USDT, Tether, or stablecoins such as mentions of tether as rope/leash or other non-crypto topics"
]

# ===== 批量判断函数（加规则预过滤，解决误判）=====
def batch_classify(texts):
    results = []
    
    for text in texts:
        if pd.isna(text) or text.strip() == "":
            results.append(False)
            continue
        
        text_lower = text.lower()
        
        # 规则预过滤：明显无关（tether比喻）
        irrelevant_patterns = [
            'invisible tether', 'tether running', 'tether behind', 'tether to', 'tethered to', 
            'tether ball', 'dog tether', 'tether him', 'tether her', 'tether around'
        ]
        if any(p in text_lower for p in irrelevant_patterns):
            results.append(False)
            continue
        
        # 规则预过滤：明显相关（加密关键词）
        related_patterns = [
            'usdt', '$usdt', 'usd tether', 'tether withdrawal', 'tether reserve', 
            'tether fraud', 'tether depeg', 'tether transparency', 'tether print'
        ]
        if any(p in text_lower for p in related_patterns):
            results.append(True)
            continue
        
        # 模糊的才进模型
        try:
            result = classifier(
                text,
                candidate_labels,
                hypothesis_template="This tweet is about {}.",
                multi_label=False
            )
            is_related = (
                result['labels'][0] == candidate_labels[0] and
                result['scores'][0] > 0.7  # 阈值0.7防误判
            )
            results.append(is_related)
        except Exception as e:
            print(f"\n⚠️ 模型错误: {e}")
            results.append(False)
    
    return results


# ===== 主处理函数 =====
def filter_usdt_tweets(input_file, output_relevant, output_irrelevant):
    print("📖 读取数据...")
    df = pd.read_csv(input_file)
    total = len(df)
    print(f"✅ 共读取 {total} 条推文\n")
    
    texts = df['text'].fillna("").tolist()
    
    print("🤖 开始批量筛选...")
    all_results = []
    
    with tqdm(total=total, desc="处理进度", unit="条") as pbar:
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = batch_classify(batch_texts)
            all_results.extend(batch_results)
            pbar.update(len(batch_texts))
    
    df['is_usdt_related'] = all_results
    
    df_relevant = df[df['is_usdt_related']].drop(columns=['is_usdt_related'])
    df_irrelevant = df[~df['is_usdt_related']].drop(columns=['is_usdt_related'])
    
    print("\n💾 保存结果...")
    df_relevant.to_csv(output_relevant, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    df_irrelevant.to_csv(output_irrelevant, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f"\n✅ 筛选完成！")
    print(f"   📊 相关: {len(df_relevant)} ({len(df_relevant)/total*100:.2f}%)")
    print(f"   📊 不相关: {len(df_irrelevant)} ({len(df_irrelevant)/total*100:.2f}%)")


if __name__ == "__main__":
    filter_usdt_tweets(
        input_file=r"D:\USDT\all_tweets_combined_updated.csv",
        output_relevant=r"D:\USDT\usdt_related_tweets.csv",
        output_irrelevant=r"D:\USDT\usdt_irrelevant_tweets.csv"
    ) 