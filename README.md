# drone-ai-agent

Agente inteligente para o desafio final de IA da INF1771: um drone autonomo
controlado por Socket TCP/IP que explora um labirinto 59 x 34 desconhecido,
coleta tesouros, evita pocos e teleportes, gerencia energia e combate outros
drones quando a relacao risco/beneficio compensa.

## Arquitetura

O projeto foi consolidado em modulos pequenos e integrados, mas a decisao
principal fica no comportamento do agente: o loop local pode rodar sem enviar
comandos quando a melhor opcao economica e esperar.

| Arquivo | Responsabilidade |
|---|---|
| `src/main.py` | CLI, perfis estrategicos, conexao, loop de partida, reset e tick adaptativo |
| `src/communication.py` | Cliente TCP/IP, protocolo do servidor, requisicoes sincronas e eventos assincronos |
| `src/observations.py` | Normalizacao e representacao estruturada das observacoes |
| `src/world_model.py` | Mapa 59 x 34, memoria da partida, inferencia Wumpus e consultas de risco |
| `src/risk.py` | Memoria temporal de ameacas e penalidades de risco |
| `src/planner.py` | BFS global, busca multi-alvo, A* com orientacao e cache de rotas |
| `src/farm.py` | Farming por utilidade economica, respawn por ponto e coletas pendentes |
| `src/combat.py` | Combate tatico, linha de tiro, misses, `CHASE`, `EVADE` e perfis |
| `src/strategy.py` | FSM comportamental, politica economica de observacao e metricas |
| `src/ai_agent.py` | Wrapper de compatibilidade para `DroneAgent` |
| `src/devkit.py` | Wrapper de compatibilidade para `GameAI` |
| `src/fuzzy.py` | Controlador fuzzy de agressividade em combate |

## Decisoes Tecnicas

- `hit` e `damage` recebidos fora da resposta de observacao ficam em
  `pending_observations` e sao mesclados na proxima observacao real.
- `steps` gera cautela temporaria, mas nao marca uma celula como perigo
  permanente. Apenas `damage` ou inimigo visivel muito perto alimentam
  `danger_cells`.
- Pocos suspeitos nunca sao atravessados. Teleporte suspeito so entra como
  fallback quando nao ha plano seguro ou o agente esta preso.
- A exploracao usa inferencia estilo Mundo de Wumpus para `breeze` e `flash`.
- O farming escolhe alvos por utilidade esperada por tempo, considerando
  deslocamento, espera ate respawn, valor aprendido e custo de coleta.
- O combate valida linha de tiro antes de disparar e interrompe sequencias de
  tiros sem `hit`.
- A fuga escolhe somente destinos comprovadamente alcancaveis por BFS/A*.
- O mapa e toda memoria especifica sao reiniciados a cada nova partida.
- `act()` nao envia mais `q` e `o` por padrao. O agente observa quando ha
  motivo: apos movimento/rotacao, tiro, bloqueio, evento assincrono, dado
  desatualizado, decisao de risco ou janela provavel de respawn.
- Durante camping/respawn, a espera local e gratuita: o agente acompanha o
  relogio local e so consulta o servidor perto da janela estimada.
- `CHASE` bloqueado nao volta imediatamente para `HUNT/CHASE`: o agente
  registra a falha, aplica cooldown para o par posicao/direcao/bloqueio,
  tenta reposicionamento lateral seguro e abandona/retoma objetivo economico
  quando nao ha progresso.
- Metricas de comandos, observacoes, tiros, coletas, loops e espera sem comando
  ficam disponiveis por `agent.metrics_snapshot()` e no resumo final.

## Requisitos

- Python 3.10+.
- Somente biblioteca padrao.
- Servidor do desafio acessivel pela porta TCP `8888`.

## Execucao

Servidor de treino com nome aleatorio:

```bash
python src/main.py
```

Host e nome especificos:

```bash
python src/main.py atari.icad.puc-rio.br MeuDrone
```

Perfil agressivo:

```bash
python src/main.py atari.icad.puc-rio.br MeuDrone --aggressive
```

Perfis explicitos:

```bash
python src/main.py --profile score
python src/main.py --profile aggressive
python src/main.py --profile safe
```

Logs detalhados:

```bash
python src/main.py --log-level debug
```

## Perfis

- `score`: perfil padrao, prioriza pontuacao, sobrevivencia e farming.
- `aggressive`: aumenta tolerancia a confronto e distancia de ataque.
- `safe`: reduz risco, foge mais cedo e e mais conservador em combate.

`--aggressive` e mantido como alias de `--profile aggressive`.

## Testes

```bash
python tests/test_offline.py
python tests/simulation_compare.py
python -m py_compile src\observations.py src\communication.py src\world_model.py src\risk.py src\planner.py src\farm.py src\combat.py src\strategy.py src\ai_agent.py src\devkit.py src\main.py
```

A suite offline usa mocks e um servidor TCP local simulado. Ela nao depende do
servidor externo.

## Comparacao Comportamental Local

`tests/simulation_compare.py` compara a politica antiga, que fazia `q+o` em
todo ciclo, com a politica atual. Resultado obtido localmente:

| Cenario | Metrica | Antes | Depois |
|---|---:|---:|---:|
| Ambiente estavel | Comandos totais | 120 | 0 |
| Ambiente estavel | Observacoes `o` | 60 | 0 |
| Ambiente estavel | Consultas `q` | 60 | 0 |
| Ambiente estavel | Pontuacao simulada | -120 | 0 |
| `steps` sem inimigo | Tiros | 0 | 0 |
| `steps` sem inimigo | Observacoes `o` | 8 | 4 |
| Inimigo sem `hit` | Tiros | 4 | 4 |
| Inimigo sem `hit` | Pontuacao simulada | -74 | -64 |
| `CHASE` bloqueado | Transicoes `CHASE/HUNT` | 15 | 0 |
| `CHASE` bloqueado | Reposicionamentos | 0 | 2 |

As taxas por minuto nesse simulador rodam sem `sleep`, entao servem apenas
para comparacao relativa entre antes/depois.

## Limitacoes Conhecidas

- O comportamento contra drones reais depende de latencia e dinamica do
  servidor oficial; localmente foram validados protocolo, planejamento,
  inferencia, farming, combate e FSM com simulacao.
- O agente nao persiste mapas entre partidas, por seguranca. Persistir um mapa
  especifico seria incorreto porque a posicao inicial e o mapa podem mudar.
- Telemetria historica e chat automatico foram deixados fora do nucleo para
  manter a solucao focada e testavel.
