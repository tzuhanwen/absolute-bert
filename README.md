# Absolute BERT

這是儲存我[碩士論文](https://hdl.handle.net/11296/t57ba6)研究程式碼的 repo，研究方向是透過對 self-attention 算法上的調整，並比較預訓練中的一些設定，提出可以增加 BERT 模型可解釋性的一種可能改進的方向。

研究主要有以下發現：
1. 如果欲讓詞向量在空間中的相對位置能代表語意並有更好的一致性，在 MLM 預訓練任務輸出的對數機率中包含截距項參數是必要的。
1. 使用完整的 cross-entropy 而非採樣過的 sampled softmax loss 才能有效地將詞向量以字詞出現的頻率對應到相應的向量大小，讓詞向量之間在長度上具有對比。詞向量有正確的長度對應可以使後續運算的句向量具有類似使用詞頻進行加權的作用，也使得 "詞向量空間的各方向分別代表不同詞義" 更加合理。

Absolute BERT 分析原本 self-attention 的算法，並參考 [Roformer](https://arxiv.org/abs/2104.09864) 對於時間嵌入的做法，調整了 dot-product attention 的運算。以規避其造成的