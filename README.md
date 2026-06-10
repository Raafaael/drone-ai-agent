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
- `HUNT` (girar procurando o atirador) só é ativado ao **levar dano** — reagir
  a `steps` girando desperdiça ações;
- **Economia de munição**: tiro custa -10 e matar (+1000) exige 10 acertos;
  após 5 tiros sem `hit` o drone desengaja por 6s em vez de sangrar pontos
  contra um alvo que desvia;
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

| Arquivo | Descrição |
|---|---|
| [main.py](main.py) | Ponto de entrada: conexão, loop principal e log em tela |
| [devkit.py](devkit.py) | Cliente TCP/IP (porta 8888) e protocolo do GameServer |
| [world_model.py](world_model.py) | Mapa, inferência/resolução lógica e A* |
| [ai_agent.py](ai_agent.py) | Máquina de estados / tomada de decisão |
| [fuzzy.py](fuzzy.py) | Controlador fuzzy de agressividade (combate) |
| [test_offline.py](test_offline.py) | Testes offline com servidor simulado |

## Como executar

```bash
# servidor de treino (padrão) com nome aleatório
python main.py

# host e nome específicos
python main.py atari.icad.puc-rio.br MeuDrone
```

Não há dependências externas — apenas Python 3 (biblioteca padrão).

O programa exibe em tela o log de todas as ações realizadas (`[ACAO]`),
mudanças de estado (`[FSM]`), planejamento (`[PLANO]`) e descobertas do mapa
(`[MAPA]`).

## Testes offline

```bash
python test_offline.py
```

Roda os testes do modelo de mundo (inferência + A*) e um teste de fumaça com
um GameServer simulado localmente, verificando que o agente explora, desvia de
obstáculos e coleta um tesouro.
