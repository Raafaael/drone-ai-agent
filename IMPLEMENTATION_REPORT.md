# Implementation Report

## Escopo

Implementacao consolidada da branch `unificada`, criada a partir da `main`,
sem merge, rebase ou cherry-pick indiscriminado das branches remotas. As
branches `origin/dante`, `origin/detonador-b` e `origin/rafael` foram usadas
como referencia tecnica.

Esta revisao corrige a primeira consolidacao, que estava excessivamente
arquitetural: os modulos existiam, mas o comportamento ainda fazia `q` e `o`
em praticamente todo ciclo, causando perda de score mesmo em espera.

Uma auditoria tecnica posterior (comparando comportamento real, nao so nomes
de arquivo/classe/estado, contra `origin/dante`, `origin/detonador-b` e
`origin/rafael`) encontrou duas regressoes concretas nesta consolidacao.
A secao "Correcoes pontuais pos-auditoria" abaixo documenta exatamente o que
foi corrigido, sem reescrever a arquitetura nem reabrir a unificacao.

## Correcoes pontuais pos-auditoria (revisao comportamental)

Rodada de correcoes pontuais sobre a arquitetura ja existente (nenhum modulo
novo, nenhum merge/cherry-pick, nenhuma mudanca na politica economica de
observacao alem do necessario para corrigir os bugs abaixo):

1. **`damage` nunca levava a `HUNT`, apesar do codigo sugerir o contrario.**
   Dois pontos escondiam o mesmo bug:
   - `combat.py:decide()` setava `self.hunt_reason = "damage"` mas os DOIS
     ramos do `if observation.damage` retornavam `"EVADE"` — a variavel de
     intencao existia, o `return "HUNT"` correspondente nunca foi escrito.
   - `strategy.py:decide_state()` tinha seu **proprio** atalho incondicional
     `if observation.damage: return "EVADE"`, que nem chegava a chamar
     `combat.decide()` — entao mesmo corrigindo so o combat.py, o bug
     continuaria de fato acontecendo.
   Correcao: `combat.py` agora retorna `"HUNT"` quando `energy > LOW_ENERGY`
   e a agressividade fuzzy (`energia x distancia assumida`) atinge o limiar
   `damage_hunt_aggr_min` do perfil ativo; `strategy.py` agora delega a
   decisao completa a `combat.decide()` em vez de fazer o curto-circuito.
   Como os limiares por perfil ja existiam (`score=0.50`, `aggressive=0.30`,
   `safe=0.70`) e sao mais altos que o de energia, o resultado pratico e:
   - `aggressive`: contra-ataca (`HUNT`) na maioria dos casos com energia
     razoavel — o perfil pensado para a partida de eliminacao volta a reagir
     a tiro em vez de so fugir;
   - `score`: contra-ataca só com energia/agressividade solida, senão evade —
     continua economico por padrao;
   - `safe`: exige agressividade bem mais alta para contra-atacar, então
     evade na esmagadora maioria dos casos — continua defensivo;
   - energia baixa (`<= LOW_ENERGY`): sempre evade, em qualquer perfil.
   Testes: `test_damage_reaction_hunts_when_strong_evades_when_weak`
   (unitario em `CombatController` para os 3 perfis + fim-a-fim via
   `DroneAgent.act()`), e a asssercao antiga que travava o bug em
   `test_combat_rules` foi corrigida para refletir o comportamento certo.

2. **`SURVEY` era um estado morto.** Existia em `STATES`, tinha handler
   (`do_survey`) e entrada no dicionario de despacho, mas nada em
   `strategy.py` jamais definia `self.state = "SURVEY"` nem
   `self.survey_turns > 0` — o estado era estruturalmente inatingivel (a
   condicao de entrada de `origin/dante`, ligada a defesa de um ponto de
   farm maduro em um modo de combate que nao existe na arquitetura atual,
   nunca foi portada). Decisao: **removido** (nao foi implementada uma nova
   condicao de entrada, para nao ampliar escopo desta rodada de correcoes
   pontuais). `SURVEY` foi retirado de `STATES`, do dicionario de despacho,
   de `decide_state()` e o metodo `do_survey`/atributo `survey_turns` foram
   apagados. Teste `test_survey_state_removed` garante que o nome nao volta
   a existir sem comportamento associado.

3. **Testes de regressao para observacao obsoleta apos acao critica.**
   O bug de respawn corrompido (observacao com `bluelight` de ANTES do
   `GRAB` sendo relida no tick seguinte) ja tinha sido corrigido em sessao
   anterior; esta rodada adiciona
   `test_no_stale_observation_reuse_after_critical_actions`, cobrindo
   explicitamente que `GRAB`, `SHOOT`, `blocked` e mudanca de celula forcam
   `_need_observation`/`_need_status` no tick seguinte (nao apenas `GRAB`).

## Arquivos Criados

- `src/communication.py`
- `src/observations.py`
- `src/risk.py`
- `src/planner.py`
- `src/farm.py`
- `src/combat.py`
- `src/strategy.py`
- `src/logger.py`
- `IMPLEMENTATION_REPORT.md`

## Arquivos Alterados

- `src/devkit.py`: passou a ser wrapper de compatibilidade.
- `src/ai_agent.py`: passou a ser wrapper de compatibilidade.
- `src/world_model.py`: ampliado para memoria completa da partida.
- `src/main.py`: refeito para perfis, reset por partida e tick adaptativo.
- `tests/test_offline.py`: suite ampliada.
- `tests/simulation_compare.py`: simulador deterministico antes/depois.
- `README.md`: documentacao atualizada.

## Componentes Aproveitados Conceitualmente

### `origin/dante`

- Preservacao de observacoes assincronas via `pending_observations`.
- Uso de `EVADE` e `CHASE`.
- Validacao de linha de tiro.
- Limite de tiros sem `hit`.
- Testes de combate e notificacoes assincronas como referencia.

`SURVEY` foi tentado, mas a condicao de entrada (defesa de camp em modo de
combate que nao existe na arquitetura atual) nunca foi portada; o estado
ficou morto ate a auditoria encontra-lo e a correcao acima remove-lo (ver
"Correcoes pontuais pos-auditoria").

### `origin/detonador-b`

- Perfis de combate e alias `--aggressive`.
- Memoria temporal de ameaca com `threat_until` e `danger_cells`.
- Penalidade de zonas perigosas.
- Scoring de fuga com destino alcancavel.
- `pending_grabs`, valor aprendido, `visit_count`, `teleport_risk`,
  `safe_exit_count` e deteccao de loops.

### `origin/rafael`

- Farming por utilidade economica.
- Pressao de exploracao inicial.
- `reachable_map`.
- Respawn individual por ponto.
- Ranking eficiente de multiplos alvos.

## Prioridades Concluidas

### Correcoes Criticas

- `hit` e `damage` assincronos sao preservados e mesclados sem serem apagados
  por nova observacao.
- Observacoes sao normalizadas em `src/observations.py`.
- Tiro e proibido quando nao ha inimigo ou quando ha parede conhecida na linha.
- Fuga usa apenas destinos alcancaveis.
- Socket tem timeout, thread de recepcao, encerramento seguro e tratamento de
  falhas.
- Estado do agente e reiniciado a cada nova partida.
- `request_sync_pair` deixou de ser a operacao padrao de todo `act()`.
- Espera por respawn/camping pode ocorrer sem nenhum comando ao servidor.
- `q` e `o` agora sao solicitados por necessidade: pos-acao, evento,
  timeout de validade, risco, bloqueio, tiro ou janela de respawn.
- Ciclos `CHASE -> HUNT -> CHASE` em perseguicao bloqueada foram tratados com
  validade temporal de inimigo, cooldown por posicao/direcao/bloqueio,
  estado real `REPOSITION`, abandono de chase sem progresso e retomada de
  plano anterior quando possivel.

### Alta Prioridade

- `visit_count`, `safe_exit_count`, `teleport_risk`.
- Memoria temporal de ameacas.
- `reachable_map` e A* com orientacao.
- Farming por utilidade, respawn por ponto, valor aprendido e `pending_grabs`.
- Estados `CHASE`, `EVADE`, `CAMP`.
- Deteccao de loops por historico de posicoes.

### Otimizacoes

- Separacao modular.
- Cache de A* invalidado por `world.version`.
- BFS multi-alvo e mapa global de alcancabilidade.
- Tick adaptativo no loop principal.
- Logger configuravel.
- Metricas de comando e comportamento expostas por `metrics_snapshot`.
- Simulacao comparativa de economia de comandos.

### Opcionais Adiados

- Telemetria historica persistente.
- Chat automatico.
- Persistencia de metricas globais.

Mapas de partidas nao sao persistidos por decisao tecnica.

## Testes Adicionados

`tests/test_offline.py` cobre:

- normalizacao de observacoes;
- `enemy`/`eneny`, luzes, `blocked`, `steps`, `damage`, `hit`;
- eventos assincronos `hit` e `damage`;
- inferencia por `breeze` e `flash`;
- visitas, risco de teleporte, saidas seguras e reset;
- BFS, A*, cache, teleporte fallback e fuga alcancavel;
- memoria temporal de ameaca;
- farming por utilidade, `pending_grabs` e valor aprendido;
- combate sem tiro por `steps`, sem tiro contra parede, `hit`, misses,
  `CHASE` e `EVADE`;
- FSM, perfis e reset;
- servidor TCP local simulado para smoke, poco, timeout e farming.
- economia de observacao durante camp;
- observacao apos movimento;
- `steps` sem tiro e sem cancelamento de rota segura;
- perfil agressivo sem aumento de polling quando nao ha inimigo.
- reproducao de `CHASE/HUNT` bloqueado;
- cooldown para mesmo bloqueio;
- expiracao de observacao antiga de inimigo;
- reposicionamento lateral;
- fuga/abandono quando flanqueamento nao existe;
- retomada de plano anterior apos chase falho;
- reacao a `damage` (`HUNT` quando forte, `EVADE` quando fraco/energia
  baixa/perfil `safe`), unitario por perfil e fim-a-fim via `DroneAgent`;
- `SURVEY` removido (nao existe mais como estado sem comportamento);
- observacao nao reaproveitada de forma obsoleta apos `GRAB`, `SHOOT`,
  `blocked` e mudanca de celula.

`tests/simulation_compare.py` cobre:

- ambiente estavel sem inimigo/item novo;
- `steps` sem inimigo;
- combate sem `hit`.

## Resultados de Validacao

Executado com sucesso:

```bash
python -m py_compile src\observations.py src\communication.py src\world_model.py src\risk.py src\planner.py src\farm.py src\combat.py src\strategy.py src\ai_agent.py src\devkit.py src\main.py
python tests/test_offline.py
python tests\simulation_compare.py
```

Resultado dos testes:

```text
Todos os testes passaram.
```

### Antes/depois da rodada de correcoes pos-auditoria

- **Antes** (suite existente, sem as mudancas desta rodada): `python
  tests/test_offline.py` **passava 100%** — nenhum teste falhava, porque
  nenhum teste ainda exercitava o caminho de `damage -> HUNT` nem a
  inatingibilidade de `SURVEY`; o bug era invisivel para a suite existente
  (a unica asssercao relacionada, em `test_combat_rules`, checava
  `combat.decide(60, ...) == "EVADE"`, ou seja, **travava o bug como se
  fosse o comportamento certo**).
- **Depois** (com as correcoes de `combat.py`/`strategy.py`, a remocao de
  `SURVEY` e os 3 testes novos/1 asssercao corrigida): suite completa
  **passa 100%** novamente, agora exercitando os dois comportamentos
  corrigidos e a garantia de que `SURVEY` nao reaparece.
- `python tests/simulation_compare.py` executado antes e depois das
  mudancas: nenhuma variacao nos cenarios de economia de comandos (a
  correcao nao mexeu na politica de observacao em si).

Simulacao comparativa executada:

| Cenario | Metrica | Antes | Depois | Variacao |
|---|---:|---:|---:|---:|
| Ambiente estavel | Comandos totais | 120 | 0 | -120 |
| Ambiente estavel | Observacoes `o` | 60 | 0 | -60 |
| Ambiente estavel | Consultas `q` | 60 | 0 | -60 |
| Ambiente estavel | Tiros | 0 | 0 | 0 |
| Ambiente estavel | Pontuacao simulada | -120 | 0 | +120 |
| `steps` sem inimigo | Comandos totais | 20 | 12 | -8 |
| `steps` sem inimigo | Observacoes `o` | 8 | 4 | -4 |
| `steps` sem inimigo | Tiros | 0 | 0 | 0 |
| `steps` sem inimigo | Pontuacao simulada | -20 | -12 | +8 |
| Inimigo sem `hit` | Comandos totais | 34 | 24 | -10 |
| Inimigo sem `hit` | Observacoes `o` | 12 | 9 | -3 |
| Inimigo sem `hit` | Consultas `q` | 12 | 5 | -7 |
| Inimigo sem `hit` | Tiros | 4 | 4 | 0 |
| Inimigo sem `hit` | Pontuacao simulada | -74 | -64 | +10 |
| `CHASE` bloqueado | Transicoes `CHASE/HUNT` | 15 | 0 | -15 |
| `CHASE` bloqueado | Tempo na mesma posicao | 8 | 0 | -8 |
| `CHASE` bloqueado | Reposicionamentos | 0 | 2 | +2 |
| `CHASE` bloqueado | Comandos enviados | 0 | 16 | +16 |

As taxas por minuto do simulador nao foram usadas como criterio absoluto,
porque os cenarios rodam sem pausas reais; elas servem apenas como comparacao
relativa.

## Limitacoes Restantes

- Nao foi validado contra o servidor externo oficial nesta etapa; a validacao
  foi feita com servidor TCP simulado e testes unitarios locais.
- Parametros de perfis podem ser calibrados com partidas reais.
- `SURVEY` foi removido por nao ter condicao de entrada funcional (ver
  "Correcoes pontuais pos-auditoria"); a FSM atual tem 10 estados, todos
  alcancaveis.
- A reducao de observacoes pressupoe que eventos assincronos relevantes
  (`hit`/`damage`) cheguem pela conexao e sejam preservados pelo cliente.

## Como Executar

```bash
python src/main.py
python src/main.py atari.icad.puc-rio.br MeuDrone --profile score
python src/main.py atari.icad.puc-rio.br MeuDrone --aggressive
python tests/test_offline.py
python tests/simulation_compare.py
```
