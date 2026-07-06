# drone-ai-agent

Agente de IA para o desafio final da disciplina INF1771 (PUC-Rio): um drone
autônomo controlado via socket TCP/IP que explora um labirinto 59×34
desconhecido, aprende onde e quando os tesouros reaparecem, evita poços e
teleportes, e reage a combate quando faz sentido — sempre priorizando
sobreviver e farmar em vez de caçar (matar um inimigo custa munição e coloca
o drone em risco; morrer encerra a participação do agente na partida).

## Arquitetura

O agente é dividido em módulos pequenos, cada um dono de uma parte do
problema. `strategy.DroneAgent` é quem amarra tudo: a cada tick, pergunta ao
mundo o que já se sabe, ao risco o quão perigosa está a vizinhança, ao farm
qual o melhor destino e ao combate o que fazer com um inimigo por perto — e
a partir disso escolhe um estado e age.

| Arquivo | Responsabilidade |
|---|---|
| `src/main.py` | Ponto de entrada: conexão, loop da partida, reinício a cada nova partida |
| `src/communication.py` | Cliente TCP/IP e protocolo do servidor (comandos, eventos assíncronos) |
| `src/observations.py` | Normalização das observações cruas do servidor |
| `src/world_model.py` | Mapa 59×34, inferência estilo Mundo de Wumpus, buscas A*/BFS |
| `src/risk.py` | Memória de ameaça e pontuação de risco de célula |
| `src/planner.py` | Fachada de busca de caminho (A* com cache, fuga, fallback de teleporte) |
| `src/farm.py` | Decide onde farmar/explorar por utilidade econômica |
| `src/combat.py` | Decide como reagir a um inimigo (atacar, perseguir, fugir, esquivar) |
| `src/strategy.py` | A máquina de estados que liga tudo isso e decide a ação de cada tick |
| `src/fuzzy.py` | Controlador fuzzy de agressividade em combate |

## Por que o agente se comporta assim

- **Sobreviver é a prioridade.** O enunciado encerra a participação do
  agente na partida quando ele morre — morrer custa muito mais que o -10 do
  placar, custa todo o farm que sobrava fazer. Por isso o combate é
  reativo (o agente não sai caçando), e uma ameaça recente pesa de verdade
  na escolha de para onde farmar/explorar, não só na hora de fugir.
- **Farm por utilidade, não por distância.** Cada ponto de item conhecido é
  avaliado por valor esperado *por segundo* até conseguir pegá-lo (viagem +
  espera até amadurecer). Isso naturalmente equilibra farmar pontos já
  maduros com explorar em busca de novos.
- **O mapa é aprendido, nunca dado.** `breeze`/`flash` só dizem que existe
  um poço/teleporte em algum vizinho; por eliminação, quando sobra um único
  candidato ele vira confirmado. Célula suspeita de poço nunca é pisada.
- **Comunicação sob demanda.** `q`/`o` só são pedidos quando algo realmente
  mudou (depois de mover/girar/atirar/pegar) ou uma leitura antiga venceu.
  Esperar parado (ex.: aguardando um respawn) não custa nenhuma mensagem ao
  servidor.
- **Perseguição com memória.** Se o caminho de uma perseguição esbarra numa
  parede, o agente tenta flanquear antes de desistir, com cooldown para não
  repetir a mesma tentativa fracassada contra a mesma parede.

## Requisitos

- Python 3.10+, apenas biblioteca padrão.
- Acesso ao servidor do desafio pela porta TCP 8888.

## Execução

```bash
# servidor de treino com nome aleatório
python src/main.py

# host e nome específicos
python src/main.py atari.icad.puc-rio.br MeuDrone

# logs mais detalhados
python src/main.py atari.icad.puc-rio.br MeuDrone --log-level debug
```

## Testes

```bash
python tests/test_offline.py
python tests/simulation_compare.py
```

A suíte offline roda sem servidor externo: usa dublês (mocks) das
respostas do servidor e um servidor TCP local simulado para os testes de
ponta a ponta (coleta, poço, farming, timeout).

`tests/simulation_compare.py` compara, de forma determinística, a política
econômica de comunicação atual com uma versão que sempre pede `q`+`o` a cada
ciclo — útil para ver o tamanho real da economia de comandos sem depender de
uma partida ao vivo.

## Limitações conhecidas

- O comportamento contra drones reais depende da latência e da dinâmica do
  servidor oficial; localmente foram validados protocolo, planejamento,
  inferência, farming, combate e FSM com simulação, não com o servidor real.
- O mapa e toda a memória aprendida são reiniciados a cada nova partida — o
  labirinto pode ser outro, então nada é persistido entre partidas.
