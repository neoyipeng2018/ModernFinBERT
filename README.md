# ModernFinBERT

A modernized financial sentiment analysis model based on the ModernBERT architecture, designed to outperform existing FinBERT variants.

## 🎯 Project Goals

- **Superior Accuracy**: Target >94% on FinancialPhraseBank (vs FinBERT's 93%)
- **Fast Inference**: <50ms per sample for production use
- **Community Impact**: Open-source release with comprehensive documentation

## 📊 Dataset

Using `neoyipeng/financial_reasoning_aggregated` from HuggingFace:
- **Total Samples**: ~31,166 (Train: 19,940, Dev: 4,992, Test: 6,234)
- **Labels**: 3-class sentiment (NEGATIVE, NEUTRAL/MIXED, POSITIVE)
- **Domain**: Financial texts with sentiment annotations

## 🚀 Development Timeline

**16-week structured development** (4 hours/week):

### Phase 1: Data Foundation (Weeks 1-3)
- ✅ **Week 1**: Data audit, label harmonization, project setup
- 🔄 **Week 2**: Statistical analysis, GPT-4 quality control
- 📅 **Week 3**: Dataset finalization and HuggingFace publication

### Phase 2: Model Development (Weeks 4-8)
- ModernBERT architecture adaptation
- Training pipeline with wandb/tensorboard
- Base and Large model variants
- Specialized domain variants

### Phase 3: Paper & Tutorial Prep (Weeks 9-12)
- Academic paper writing
- Tutorial development
- Conference submission

### Phase 4: Community Building (Weeks 13-16)
- Public launch and documentation
- Community engagement
- Production deployment guides

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/ModernFinBERT.git
cd ModernFinBERT

# Install dependencies
pip install -r requirements.txt

# Start with Phase 1 data exploration
jupyter notebook Data.ipynb
```

## 📁 Project Structure

```
ModernFinBERT/
├── data/                    # Financial text datasets and cleaning scripts
├── models/                  # ModernBERT model variants and weights  
├── scripts/                 # Training and evaluation scripts
├── notebooks/               # Jupyter notebooks for experimentation
├── docs/                    # Documentation and tutorials
├── Data.ipynb              # Phase 1 implementation guide
├── plan.md                 # Detailed 16-week development plan
└── CLAUDE.md               # Development guidance for Claude Code
```

## 🎯 Success Metrics

- [x] **Data Foundation**: Clean, high-quality dataset
- [ ] **Model Performance**: >94% FinancialPhraseBank accuracy
- [ ] **Inference Speed**: <50ms per sample
- [ ] **Community Adoption**: 100+ GitHub stars in first month
- [ ] **Academic Impact**: Conference tutorial acceptance
- [ ] **Industry Usage**: 3+ companies in production

## 🤝 Contributing

This project follows a structured development timeline. See `CLAUDE.md` for detailed guidance on working with this codebase.

## 📄 License

[Add your preferred license here]

## 🔗 Links

- **Dataset**: [neoyipeng/financial_reasoning_aggregated](https://huggingface.co/datasets/neoyipeng/financial_reasoning_aggregated)
- **Base Model**: [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- **Development Plan**: [plan.md](./plan.md)

---

**Status**: 🚧 Phase 1 (Data Foundation) - Week 1 implementation ready