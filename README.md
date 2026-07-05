# drone-ai-agent

Agente inteligente para o **Desafio Final de IA (INF1771 – PUC-Rio)**: um drone
autônomo controlado via Socket TCP/IP que explora um labirinto 59×34
desconhecido, coleta tesouros, evita poços e teleportes e combate outros
drones.

## Técnicas de IA utilizadas (e onde aparecem na disciplina)

- **Máquina de Estados Finitos** *(IA em Jogos / Máquinas de Estado)* —
  estados `EXPLORE`, `GRAB`, `ATTACK`, `HUNT`, `RECHARGE` e `FLEE`
  ([ai_agent.py](ai_agent.py));
- **Agentes Lógicos / inferência (Mundo de Wumpus)** *(Introdução aos
  Agentes Lógicos)* — dos sensores `breeze`/`flash` o agente deduz células
  suspeitas e comprovadamente seguras; por **resolução** (eliminação de
  candidatos), confirma a posição exata de poços e teleportes e calcula um
  risco relativo por contagem de evidências ([world_model.py](world_model.py));
- **Busca heurística A\* ciente de rotação** *(Busca Heurística /
  Pathfinding e Controle de Movimento)* — cada giro de 90° custa 1 ação
  (caminhos retos são preferidos), células não exploradas são penalizadas e
  células suspeitas de poço são **proibidas**;
- **Lógica Fuzzy** *(Lógica Fuzzy)* — controlador Sugeno em
  [fuzzy.py](fuzzy.py): energia × distância do inimigo → *agressividade*
  em [0,1], que decide entre `ATTACK`, `HUNT` e `FLEE` sem limiares rígidos;
- **Exploração por fronteira com patrulha** *(Waypoints / Tomada de Decisão
  tática)* — prioriza itens conhecidos, depois a fronteira do desconhecido
  e, com o mapa esgotado, patrulha os pontos de coleta mais antigos (itens
  reaparecem).

## Estratégia de pontuação: farming de respawns

Os itens do servidor **reaparecem** após coletados. Tesouros únicos somam
pouco (~2-3 mil); a pontuação alta vem de tratar cada ponto de tesouro
como recurso renovável:

1. **FARM** — prioridade máxima: voltar a pontos de item conhecidos que já
   devem ter reaparecido (estimativa adaptativa do tempo de respawn,
   aprendida observando a luz ao ficar sobre o ponto);
2. **EXPLORAR** — sem ponto "maduro", expandir a fronteira (descobre novos
   pontos para engrossar o circuito);
3. **ACAMPAR** — mapa esgotado: parar **em cima** do ponto mais antigo e
   esperar a luz acender (esperar é grátis; pegar custa 1 ação e rende
   +1000);
4. **ESPERAR** — nada alcançável: não gastar ações.

A coleta é disparada pela **luz observada** (verdade do servidor), não por
cooldown — se o item reapareceu, o drone pega na hora.

## Decisões de projeto (desempenho)

- Célula suspeita de **poço nunca é pisada** (-1000 e fim de jogo não
  compensam nenhum tesouro); suspeita de **teleporte** só é atravessada como
  último recurso, quando não existe rota segura (teleporte não mata);
- Powerup só é coletado com energia ≤ 70 (com energia cheia o item seria
  desperdiçado, e a tentativa custa -5);
- **Economia de munição**: tiro custa -10 e matar (+1000) exige 10 acertos;
  após 5 tiros sem `hit` o drone desengaja por 6s em vez de sangrar pontos
  contra um alvo que desvia;
- **Memória de ameaça com dois níveis de confiança**: `steps` (áudio) é um
  sensor ambiente — não indica direção nem garante proximidade real, e o
  drone passa longos períodos parado sobre pontos de farm, então tratá-lo
  como "perigo naquela célula" envenenaria justamente o melhor ponto
  conhecido por ruído de fundo. Por isso `threat_until` (janela de cautela
  que só decide **quando** vale a pena tentar um scan) reage a qualquer sinal
  (`steps`/`damage`/`enemy#N`), mas `danger_cells` (usada para **penalizar**
  farm/fuga) só registra evidência forte: dano realmente recebido ou inimigo
  a queima-roupa (`enemy_dist` ≤ 3);
- **Farm/exploração evitam zona de risco recente**: `_target_score` penaliza
  células perto de uma detecção forte de inimigo (decai com tempo e
  distância), reduzindo a chance de repetir farm num ponto onde o drone
  levou tiro há pouco;
- **Fuga direcionada**: `_flee_score` não maximiza só a distância da posição
  atual — prioriza alvos que aumentem a distância em relação à última célula
  de ameaça conhecida (fugir "para longe de onde eu estava" pode passar perto
  de onde o tiro veio);
- **`HUNT` reativo e proativo sem custo de rota**: além do `HUNT` reativo (ao
  levar tiro), passos persistentes (não um blip isolado) com energia alta
  também disparam uma busca curta pelo inimigo (cooldown de 20s, exige
  agressividade alta). Como `HUNT` só gira no lugar, a transição
  `EXPLORE → HUNT → EXPLORE` preserva o caminho de farm/exploração em
  andamento em vez de descartá-lo — interromper e replanejar do zero a cada
  scan reduziria a taxa de coleta sem ganho tático nenhum;
- **Farm com múltiplas coletas pendentes**: coletas em sequência rápida em
  pontos vizinhos (antes da primeira confirmar no score) são resolvidas
  independentemente, sem perder a amostra de aprendizado de valor de nenhuma
  delas;
- **Dados frescos ou nada**: se status/observação não chegam dentro do
  timeout, o tick é descartado — agir com dados velhos atribuiria sensores à
  célula errada e poderia marcar como "seguro" o vizinho de um poço. O
  timeout cresce adaptativamente com servidor lento;
- **Alvos por BFS multi-alvo**: uma única varredura garante encontrar
  qualquer alvo alcançável (sem colapsar em passeio aleatório); o A* refina
  o caminho minimizando giros;
- **Ficar parado é grátis**: sem alvos alcançáveis, o drone para de andar
  (ações custam -1) e espera respawn de itens, em vez de vagar;
- Status e observação são requisitados **em paralelo** (metade da latência) e
  há detecção de travamento (comando de andar sem efeito → célula marcada
  como bloqueada e rota replanejada);
- O modelo de mundo é **zerado a cada nova partida** (o mapa pode mudar).

## Estrutura

```
drone-ai-agent/
├── src/            # codigo do agente
├── tests/          # testes offline (servidor simulado)
└── Instrucoes/     # enunciado do trabalho
```

| Arquivo | Descrição |
|---|---|
| [src/main.py](src/main.py) | Ponto de entrada: conexão, loop principal e log em tela |
| [src/devkit.py](src/devkit.py) | Cliente TCP/IP (porta 8888) e protocolo do GameServer |
| [src/world_model.py](src/world_model.py) | Mapa, inferência/resolução lógica e A* |
| [src/ai_agent.py](src/ai_agent.py) | Máquina de estados / tomada de decisão |
| [src/fuzzy.py](src/fuzzy.py) | Controlador fuzzy de agressividade (combate) |
| [tests/test_offline.py](tests/test_offline.py) | Testes offline com servidor simulado |

## Como executar

```bash
# servidor de treino (padrão) com nome aleatório
python src/main.py

# host e nome específicos
python src/main.py atari.icad.puc-rio.br MeuDrone
```

Não há dependências externas — apenas Python 3 (biblioteca padrão).

O programa exibe em tela o log de todas as ações realizadas (`[ACAO]`),
mudanças de estado (`[FSM]`), planejamento (`[PLANO]`) e descobertas do mapa
(`[MAPA]`).

## Testes offline

```bash
python tests/test_offline.py
```

Roda os testes do modelo de mundo (inferência + A*), lógica fuzzy, economia de
tiro, memória de ameaça/perigo, fuga direcionada, farm com coletas pendentes
concorrentes, e testes de fumaça com um `GameServer` simulado localmente
(incluindo um inimigo simulado), verificando que o agente explora, desvia de
obstáculos, foge/ataca de forma coerente e coleta/farma itens.
