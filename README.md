# Detecção de Objetos Proibidos em Zonas Restritas

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)
![Backend](https://img.shields.io/badge/OpenCV-4.x-blue.svg)

Este projeto implementa uma solução de visão computacional para detectar objetos proibidos que permanecem por um tempo determinado em zonas restritas pré-definidas em um feed de vídeo. É ideal para aplicações de segurança e monitoramento automatizado.

<!-- Coloque um GIF aqui para demonstrar o projeto em ação! -->
<!-- ![Demonstração](./demo.gif) -->

---

## 📝 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Começando](#começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Uso](#uso)
  - [Configuração](#configuração)
  - [Execução](#execução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## 📖 Sobre o Projeto

O objetivo principal é identificar quando objetos específicos, como facas (`knife`), tesouras (`scissors`) ou garrafas (`bottle`), entram e permanecem em áreas de monitoramento. O sistema utiliza um modelo de detecção de objetos YOLOv8, mas pode ser adaptado para outros modelos.

A lógica de alerta não é baseada apenas na detecção, mas também no **tempo de permanência (dwell time)**. Um alerta só é gerado se um objeto proibido for rastreado dentro de uma zona restrita por um período superior ao configurado, evitando falsos positivos de objetos que apenas cruzam a área rapidamente.

---

## ✨ Funcionalidades

-   **Detecção em Tempo Real**: Processa feeds de vídeo de webcam, arquivos ou streams RTSP.
-   **Zonas Proibidas Configuráveis**: Defina múltiplos polígonos na imagem como zonas restritas.
-   **Alerta por Tempo de Permanência**: Dispara eventos apenas quando um objeto permanece na zona por um tempo mínimo.
-   **Rastreamento Simples por IoU**: Acompanha objetos entre frames para calcular o tempo de permanência.
-   **Registro de Eventos**: Salva todos os alertas em um arquivo CSV (`data/eventos_proibidos.csv`) com timestamp e detalhes da detecção.
-   **Benchmarking de Modelos**: Notebooks na pasta `jupyter/` para treinar e comparar o desempenho de diferentes versões do YOLO (v8, v9, v10).

---

## 🚀 Começando

Siga estas instruções para colocar o projeto em execução no seu ambiente local.

### Pré-requisitos

-   Python 3.8 ou superior
-   `pip` (gerenciador de pacotes do Python)
-   (Opcional) Uma GPU NVIDIA com CUDA para melhor desempenho de inferência.

### Instalação

1.  **Clone o repositório:**
    ```sh
    git clone https://github.com/SEU_USUARIO/computerVision-forbiddenObjects.git
    cd computerVision-forbiddenObjects
    ```

2.  **Crie e ative um ambiente virtual:**
    ```sh
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **(Opcional) Configure variáveis de ambiente:**
    Copie o arquivo de exemplo `.env.example` para um novo arquivo chamado `.env` e preencha as variáveis, se necessário para seus scripts de treino ou outros.
    ```sh
    copy .env.example .env
    ```
---

## Usage

O script principal `main.py` é altamente configurável, tanto por argumentos de linha de comando quanto por constantes no início do arquivo.

### Configuração

Antes de executar, você pode querer ajustar os seguintes parâmetros dentro de `main.py`:

-   `PROHIBITED_CLASS_NAMES`: Conjunto de classes de objetos a serem considerados proibidos.
    ```python
    PROHIBITED_CLASS_NAMES = {"knife", "scissors", "bottle"}
    ```
-   `ZONES_NORM`: Lista de polígonos que definem as zonas restritas. As coordenadas são normalizadas (de 0 a 1), facilitando a adaptação a diferentes resoluções de vídeo.
    ```python
    ZONES_NORM = [
        {
            "name": "No-Blade Zone 1",
            "polygon": [(0.05, 0.60), (0.60, 0.60), (0.60, 0.95), (0.05, 0.95)]
        },
    ]
    ```
-   `DWELL_SECONDS`: Tempo mínimo (em segundos) que um objeto deve permanecer na zona para disparar um alerta.

### Execução

Use o terminal para rodar a detecção.

-   **Para usar a webcam (padrão):**
    ```sh
    python main.py --weights yolov8m.pt
    ```
    *Pressione `q` ou `ESC` para fechar a janela de visualização.*

-   **Para usar um arquivo de vídeo:**
    ```sh
    python main.py --weights yolov8n.pt --source "caminho/para/seu/video.mp4"
    ```

-   **Para usar um stream RTSP:**
    ```sh
    python main.py --weights yolov8s.pt --source "rtsp://seu_stream_url"
    ```

### Análise e Treinamento

A pasta `jupyter/` contém vários notebooks para tarefas mais avançadas:
-   `yolo.pipeline.ipynb`: Pipeline para treinar e avaliar modelos YOLO.
-   `Benchmark_Visao/`: Contém resultados e configurações de benchmarks comparando YOLOv8, v9 e v10. Explore esta pasta para ver qual modelo teve o melhor desempenho.

---

## 📂 Estrutura do Projeto

```
.
├── main.py                 # Script principal para detecção em tempo real
├── requirements.txt        # Dependências do projeto
├── jupyter/                # Notebooks para experimentação, treino e benchmarks
│   ├── Benchmark_Visao/    # Resultados dos testes com YOLO v8, v9, v10
│   └── ...
└── src/                    # Módulos Python (código fonte modularizado)
    ├── configs/
    ├── data/
    ├── models/
    └── utils/
```

---

## 🤝 Contribuindo

Contribuições são o que tornam a comunidade de código aberto um lugar incrível para aprender, inspirar e criar. Qualquer contribuição que você fizer será **muito apreciada**.

1.  Faça um *Fork* do Projeto
2.  Crie sua *Feature Branch* (`git checkout -b feature/SuaFeatureIncrivel`)
3.  Faça o *Commit* de suas mudanças (`git commit -m 'Adiciona SuaFeatureIncrivel'`)
4.  Faça o *Push* para a *Branch* (`git push origin feature/SuaFeatureIncrivel`)
5.  Abra um *Pull Request*

---

## 📄 Licença

Distribuído sob a Licença MIT. Veja `LICENSE` para mais informações.
