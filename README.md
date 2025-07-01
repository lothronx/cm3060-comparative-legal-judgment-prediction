# Comparative Legal Judgment Prediction Using Statistical and Embedding-based Models

**Course:** CM3060 Natural Language Processing  
**Author:** Yue Wu  
**Date:** June 30, 2025

## Project Overview

This project presents a comprehensive comparative analysis of traditional statistical models and modern embedding-based deep learning approaches for Legal Judgment Prediction (LJP), specifically focusing on criminal charge classification. The study evaluates the effectiveness of TF-IDF + SVM versus BERT-based dual encoder architectures in predicting criminal charges from case descriptions using the CAIL2018 dataset.

## Context and Background

Legal Judgment Prediction (LJP) represents a critical intersection of legal intelligence and artificial intelligence, aiming to forecast judicial outcomes from textual case descriptions. This field has evolved from early quantitative analyses in the 1980s to sophisticated machine learning approaches, driven by the increasing digitization of legal documents and advances in Natural Language Processing (NLP) technologies.

### Key Research Questions

This project addresses three core objectives:

1. **Assess Trade-offs Between Model Complexity and Performance**: Investigating whether advanced deep learning models significantly outperform traditional statistical methods in terms of accuracy and robustness for legal text classification.

2. **Explore Domain-Specific Knowledge Integration**: Evaluating how incorporating legal domain knowledge (such as legal articles) impacts model performance across both traditional and deep learning approaches.

3. **Contribute to Practical LJP System Design**: Providing empirical insights into the strengths and limitations of statistical and embedding-based models to guide future research and development in LJP.

## Dataset

The project utilizes the **CAIL2018 dataset**, a publicly available benchmark dataset for Chinese legal judgment prediction. This dataset is particularly significant as one of the few large-scale, well-annotated, publicly accessible datasets that include real-world judicial case descriptions with structured judgment outcomes.

### Dataset Characteristics:

- **Source**: Chinese AI and Law (CAIL) Challenge
- **Scale**: Approximately 2.68 million criminal legal documents
- **Coverage**: 183 different charges under China's Criminal Law
- **Subsets**:
  - CAIL2018-Small: ~200,000 samples (154,592 training, 17,131 validation, 32,508 test)
  - CAIL2018-Large: >1.5 million samples

### Data Preprocessing

For this study, the dataset was reduced to focus on the five most frequently occurring criminal charges:

- 盗窃 (Theft)
- [走私、贩卖、运输、制造]毒品 (Drug-related crimes)
- 故意伤害 (Intentional injury)
- 抢劫 (Robbery)
- 诈骗 (Fraud)

The preprocessing pipeline included:

1. **Text Cleaning**: Removal of URLs, mentions, hashtags, and non-Chinese characters
2. **Tokenization**: Using jieba library for Chinese word-level tokenization
3. **Stopword Removal**: Filtering common Chinese stopwords
4. **Label Encoding**: Converting text labels to numerical format
5. **Dataset Flattening**: Converting multi-label cases to single-label instances

## Methodology

### Model 1: Statistical Baseline (TF-IDF + SVM)

The baseline model employs a traditional machine learning approach combining:

- **Text Representation**: TF-IDF vectorization with max_features=5000 and ngram_range=(1,2)
- **Classifier**: Linear kernel Support Vector Machine (SVM) with C=1.0
- **Advantages**: Simplicity, interpretability, computational efficiency
- **Use Case**: Ideal for well-structured, balanced datasets with clear linguistic patterns

### Model 2: Embedding-based Deep Learning (BERT Dual Encoder)

The advanced model utilizes a sophisticated neural architecture:

- **Base Model**: BERT-base-Chinese for encoding
- **Architecture**: Dual encoder framework processing both case facts and legal statutes
- **Training**: Contrastive learning with InfoNCE loss function
- **Knowledge Integration**: Legal statute descriptions for domain-specific context
- **Advantages**: Captures semantic nuances, handles complex text relationships

#### Technical Details:

- **Shared Encoder**: BERT model encodes both facts and statutes into common semantic space
- **Loss Function**: Symmetric contrastive loss (InfoNCE/NT-Xent)
- **Training Configuration**:
  - Batch sizes: 16 (training), 32 (evaluation)
  - Optimizer: AdamW with learning rate 2e-5
  - Scheduler: Linear warmup (10%) + linear decay
  - Epochs: 10 with early stopping

## Results and Performance Analysis

### Quantitative Results

| Model             | Accuracy | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| ----------------- | -------- | -------- | ------------------ | --------------- | ----------- |
| TF-IDF + SVM      | 96.00%   | 95.73%   | 96.00%             | 96.00%          | 96.00%      |
| BERT Dual Encoder | 97.23%   | 96.83%   | 97.23%             | 97.23%          | 97.23%      |

### Key Findings

1. **Marginal Performance Difference**: Both models demonstrate strong performance, with the embedding model showing only slight improvements (1.23% accuracy gain).

2. **Semantic Understanding**: The BERT model better distinguishes between semantically similar crimes (theft vs. robbery), making fewer confusion errors (79 vs. 129 misclassifications).

3. **Drug Crime Classification**: Both models achieved near-perfect performance on drug-related crimes, suggesting clear linguistic patterns distinguish these cases.

4. **Computational Trade-offs**: While the embedding model performs slightly better, it requires significantly more computational resources and training time.

### Confusion Matrix Analysis

Both models show similar confusion patterns, primarily between:

- **Theft (盗窃) ↔ Robbery (抢劫)**: Expected due to semantic similarity (difference lies in use of violence)
- **Theft (盗窃) ↔ Fraud (诈骗)**: Some overlap in property-related crimes

The embedding model demonstrates better discrimination between these semantically close categories.

## Future Work and Limitations

### Current Limitations:

1. **Dataset Scope**: Limited to five most frequent charges
2. **Language Specificity**: Chinese legal system focus
3. **Computational Constraints**: Limited hyperparameter optimization for BERT model
4. **Evaluation Scope**: Single domain evaluation

### Future Directions:

1. **Extended Evaluation**: Test on more challenging datasets with larger class imbalances
2. **Cross-lingual Analysis**: Extend to other legal systems and languages
3. **Advanced Architectures**: Explore transformer variants and legal-specific pre-trained models
4. **Interpretability**: Develop explanation mechanisms for embedding-based models
5. **Real-world Deployment**: Test in actual legal information systems

## Contributing to Legal AI

This work contributes to the broader field of legal artificial intelligence by:

1. **Empirical Evidence**: Providing concrete evidence about when traditional vs. modern approaches are most effective
2. **Methodology**: Establishing a robust comparison framework for LJP models
3. **Practical Guidance**: Offering decision criteria for model selection in legal applications
4. **Resource Efficiency**: Demonstrating that simpler models can be competitive under certain conditions

## References

The project builds upon extensive research in legal judgment prediction, natural language processing, and machine learning. Key references include recent advances in BERT-based legal text analysis, traditional statistical methods in legal informatics, and domain-specific knowledge integration approaches.

## License and Citation

This project is developed for educational purposes as part of the CM3060 Natural Language Processing course. The CAIL2018 dataset is used under its original license terms. If you use this work, please cite appropriately and respect the original dataset licensing.

---

_For detailed implementation, complete results, and technical specifications, please refer to the accompanying Jupyter notebook._
