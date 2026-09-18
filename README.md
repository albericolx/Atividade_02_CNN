# Atividade 02 - Arquiteturas de Redes Neurais Convolucionais (CNNs) 🧠🖼️

Este repositório contém a implementação da **Atividade 02** da disciplina de **Tópicos Avançados de IA** (UFRPE). O objetivo principal é demonstrar domínio sobre arquiteturas de redes neurais CNN e avaliar empiricamente como modificações estruturais impactam o treinamento, o custo computacional e a acurácia de inferência.

## 📄 Artigo Base e Dataset
Toda a fundamentação, treinamento e estudo de modificações (ablação) deste projeto são baseados no seguinte artigo científico:
* **Artigo**: *Empirical Ablation and Ensemble Optimization of a Convolutional Neural Network for CIFAR-10 Classification* (arXiv, 2026). [Link](http://arxiv.org/abs/2604.23861v1).
* **Dataset**: **CIFAR-10** (Krizhevsky et al.), composto por imagens coloridas de 32x32 pixels focadas em 10 classes do cotidiano (animais, veículos).

## 🛠️ Modificações Arquiteturais (Estudo de Ablação)
Partindo de um modelo CNN Baseline (duas camadas convolucionais + ReLU + MaxPooling), foram aplicadas e analisadas as seguintes 3 modificações cumulativas:
1. **Modificação 1 (Regularização)**: Adição de uma camada de **Dropout (0.5)** para prevenção de *overfitting*.
2. **Modificação 2 (Ativação)**: Troca da função de ativação padrão (ReLU) por **Leaky ReLU**, garantindo o fluxo de gradientes negativos.
3. **Modificação 3 (Profundidade)**: Inclusão de uma **3ª Camada Convolucional**, aumentando a extração de características (features) de alto nível.

## 📁 Estrutura do Repositório (GitFlow)

O projeto segue a estrutura padrão do workspace da disciplina:
```text
Atividade_02_CNN/
├── code/               # Scripts Pytorch (main.py) e resultados gerados (gráficos e logs)
├── papers/             # Código fonte do artigo técnico em LaTeX (Sincronizado com Overleaf)
├── core/               # Scripts de infraestrutura (como busca de artigos no arXiv)
├── CONTEXT.md          # Contexto local da pasta
├── ROADMAP.md          # Passo a passo de execução
├── ISSUES.md           # Registros de dependências e bugs
└── README.md           # Documentação principal
```

## 🚀 Como Executar

1. **Clone o repositório**
   ```bash
   git clone https://github.com/albericolx/Atividade_02_CNN.git
   cd Atividade_02_CNN/code
   ```

2. **Instale as dependências**
   Certifique-se de ter o Python instalado (3.10+ recomendado) e execute:
   ```bash
   pip install torch torchvision matplotlib
   ```

3. **Execute o Treinamento e Inferência**
   O script cuidará automaticamente de fazer o download do dataset CIFAR-10, treinar as 4 variações da CNN e plotar os gráficos:
   ```bash
   python main.py
   ```
   *Os gráficos comparativos serão gerados dentro da pasta `results/`.*

## 📊 Resumo dos Resultados Obtidos

| Modelo | Tempo de Treino (s) | Acurácia (%) | Tamanho (Params) | Loss Final |
| :--- | :--- | :--- | :--- | :--- |
| **Base** | 42.61 | 51.50 | 532202 | 1.2888 |
| **Mod 1 (Dropout)** | 47.27 | 52.60 | 532202 | 1.3905 |
| **Mod 2 (LeakyReLU)** | 49.67 | 53.65 | 532202 | 1.3225 |
| **Mod 3 (3 Camadas)** | 69.25 | 53.30 | 288554 | 1.3429 |

---
**Autor**: Alberico ([albericolx](https://github.com/albericolx))  
**Disciplina**: Tópicos Avançados de Inteligência Artificial - UFRPE
