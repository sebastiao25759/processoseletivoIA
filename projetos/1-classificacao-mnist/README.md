# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Sebastião Araujo Rodrigues**

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo implementado em `train_model.py` consiste em uma Rede Neural Convolucional (CNN) para classificação de imagens. A arquitetura é composta por **3 blocos convolucionais**, cada um formado por camadas `Conv2D`, `BatchNormalization` e `MaxPooling2D`, responsáveis pela extração progressiva das características das imagens e redução da dimensionalidade espacial. Também foi utilizado `Dropout` para reduzir o risco de overfitting durante o treinamento.

Após os blocos convolucionais, a saída é achatada (`Flatten`) e conectada a camadas totalmente conectadas (`Dense`), finalizando com uma camada de saída utilizando a função de ativação `Softmax` para classificação.

Durante o treinamento foi utilizada uma estratégia de validação com conjunto de validação separado, além de `EarlyStopping`, que interrompe o treinamento quando a perda de validação deixa de melhorar por um determinado número de épocas, restaurando os melhores pesos encontrados.

Uma escolha de hiperparâmetro importante foi a utilização de `Dropout` com taxa de 0.5 antes da camada de saída. Essa configuração foi adotada para reduzir o overfitting, evitando que a rede ficasse excessivamente dependente de determinados neurônios durante o treinamento e melhorando a capacidade de generalização para imagens não vistas.

Além disso, foi utilizado EarlyStopping com restauração dos melhores pesos (`restore_best_weights=True`), permitindo interromper o treinamento quando a perda de validação deixasse de apresentar melhorias e mantendo o estado do modelo com melhor desempenho encontrado durante o treinamento.

### 2️⃣ Bibliotecas Utilizadas

- Python 3.10
- TensorFlow 2.21
- Keras 3.12.3
- NumPy 2.2.6

### 3️⃣ Técnica de Otimização do Modelo

A otimização foi realizada por meio da **Dynamic Range Quantization** durante a conversão do modelo TensorFlow/Keras (`model.h5`) para o formato TensorFlow Lite (`model.tflite`), utilizando a otimização padrão do TensorFlow Lite (`tf.lite.Optimize.DEFAULT`).

Essa técnica reduz o tamanho do modelo ao aplicar quantização nos pesos da rede, reduzindo a precisão das representações numéricas utilizadas no armazenamento. Durante a inferência, as ativações permanecem em ponto flutuante, permitindo uma redução significativa no tamanho do modelo e melhor eficiência computacional, mantendo um bom nível de precisão.

A técnica foi escolhida por apresentar um melhor equilíbrio entre redução de tamanho e desempenho. Durante os testes, a alternativa de **Full Integer Quantization** apresentou uma queda significativa na acurácia, atingindo aproximadamente 13%, enquanto a Dynamic Range Quantization manteve a acurácia de validação em 99%.

### 4️⃣ Resultados Obtidos

- **Acurácia no conjunto de validação:** 99%

- **Tamanho real do modelo original (`model.h5`):** 2,84 MB

- **Tamanho real do modelo otimizado (`model.tflite`):** 0,24 MB

A aplicação da Dynamic Range Quantization reduziu o tamanho do modelo de aproximadamente 2,84 MB para 0,24 MB, representando uma redução significativa no armazenamento necessário para execução em dispositivos Edge, mantendo a acurácia de validação em 99%.

### 5️⃣ Comentários Adicionais (Opcional)

A principal dificuldade encontrada durante o desenvolvimento ocorreu ao executar o projeto localmente utilizando **Python 3.11**. Nessa configuração, as versões utilizadas das bibliotecas TensorFlow e Keras apresentavam incompatibilidade ao carregar o arquivo `model.h5` salvo pelo Keras, resultando em um erro durante a desserialização da camada `InputLayer`, onde argumentos como `batch_shape` e `optional` não eram reconhecidos.

Para investigar a causa do problema, realizei testes na minha máquina, utilizando diferentes versões do Python, TensorFlow e Keras. Esses testes permitiram identificar que o erro estava relacionado à compatibilidade entre essas versões. Após ajustar as versões utilizadas no ambiente local, foi possível executar corretamente o carregamento do modelo e validar o fluxo completo de treinamento, conversão e inferência.

Também foi realizada uma tentativa de utilizar a técnica de **Full Integer Quantization** para obter uma otimização ainda maior do modelo. Entretanto, durante a execução dos testes do workflow, a acurácia do modelo caiu para aproximadamente **13%**, fazendo com que ele não atendesse aos critérios mínimos de desempenho estabelecidos. Por esse motivo, optou-se pela **Dynamic Range Quantization**, que apresentou uma redução significativa no tamanho do modelo, preservando uma acurácia de validação de **99%** e garantindo a aprovação nos testes automatizados.

Além disso, a utilização de Batch Normalization, Dropout, Early Stopping e da técnica de Dynamic Range Quantization contribuiu para obter um modelo com boa capacidade de generalização, menor tamanho e adequado para inferência em TensorFlow Lite.

### 6️⃣ Exemplo de Inferência

```text
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

**Comentário**

Nas cinco amostras testadas, o modelo classificou corretamente todos os dígitos, obtendo 100% de acerto nesse conjunto de inferência. Esse resultado demonstra que o modelo convertido para TensorFlow Lite manteve seu desempenho após a aplicação da Dynamic Range Quantization. Entretanto, essa avaliação representa apenas uma pequena amostra, sendo a acurácia de validação a métrica mais representativa para avaliar o desempenho geral do modelo.