# Year 5: Developing a Novel Multiple Linear Regression Model To Optimize Honey Bee Gut Immunity Using a Lactic Acid Bacteria Probiotic Mixture (2022)

## Scheda

| Campo | Valore |
|---|---|
| Anno | 2022 (Regeneron ISEF, Atlanta; progetto ANIM011) |
| Premi (tutti) | **First Award of $5,000** (Animal Sciences) + **EU Contest for Young Scientists Award** (rappresentare ISEF e gli USA a EUCYS 2022, Leiden, Paesi Bassi) [web: societyforscience.org/press-release/regeneron-isef-full-awards-2022]. Premi precedenti dello stesso filone: ISEF 2021 (progetto ANIM007, "Year 4") First Award of $5,000 + University of Arizona Renewal Tuition Scholarship [dataset locale ISEF 2021, ProjectId 20499]. ISEF 2019 ("Year Two") e ISEF 2020 ("Year Three"): nessun premio registrato. |
| Categoria | Animal Sciences (ANIM) |
| Finalista/i | Varun Madan |
| Scuola | Lake Highland Preparatory School |
| Luogo | Orlando, Florida, USA [web: inverse.com "Orlando Teen Varun Madan…"] |
| Mentore / laboratorio | non trovato (nessuna fonte accessibile nomina un mentore, un'università o un laboratorio; gli abstract 2019-2020 usano "our project"/"we", che suggerisce un supervisore o un apicoltore di supporto, ma nessun nome è reperibile) |
| Codice pubblico | nessun codice pubblico trovato. L'account GitHub dell'autore (github.com/madanva, "Varun Madan — cs @ stanford", 24 repository, tutte 2023-2026, nessuna su api/Nosema/MLR/iOS) non contiene il modello né l'app [web: github.com/madanva] |
| Pubblicazioni | non trovate (nessun lavoro peer-reviewed, preprint o brevetto a nome dello studente su api/Nosema reperibile con i canali disponibili) |
| Tipo di ricerca | Sperimentale (studio di campo su alveari + studi in gabbia) + modellistica/ML applicata (regressione lineare multipla con aggiornamento continuo) + prototipo software (app iOS); progetto longitudinale di 5 anni |

## Abstract ufficiale (EN)

[abstract] "Honeybees are essential to society, performing over 80 percent of worldwide pollination. However, commercial hive populations have been decreasing at an alarming rate in the United States. Research throughout the first three years of the project revealed that treating the hives with a specialized lactic acid bacteria probiotic mixture containing B. infantis, B. bifidium, and L. kunkeei significantly reduced the counts of the harmful gut parasite Nosema ceranae. An existing Scikit-learn multiple linear regression model was used last year to discern the optimal therapeutic index of the previously proven bacterial treatment for any given initial concentration. However, a major problem with this (and other existing machine learning alternatives) was that the model would not minimize the sum of the residuals unless the explicit training command was given. To this end, a novel multiple linear regression model was developed with continual-learning algorithms, and its accuracy was compared with experimental values & existing ML alternatives. The results showed an extremely significant improvement in output accuracy, as the margin of error in predicted Nosema concentration significantly decreased at every treatment dosage value throughout the experiment. Additionally, when compared to existing machine learning libraries, the model's continual-learning algorithms predicted output values with significantly more accuracy (peak of 92.8%) without sacrificing significant processing time. After continuing to integrate this multiple linear regression model into a user-friendly iOS application, farmers will potentially have the ability to determine the ideal dosage of the bacterial treatment just moments after diagnosing their hives' Nosema concentrations."

Versione breve pubblicata su ProjectBoard (ANIM011 "Optimizing Bee Gut Immunity with a Novel MLR Model") [web: projectboard.world]: "After training, there was an insignificant difference between the novel MLR model's predictions and experimental values, and the MLR was significantly more accurate (peak of 92.8%) than existing ML alternatives."

## Problema e domanda di ricerca

Il problema di fondo è il declino delle colonie commerciali di *Apis mellifera* negli USA e il ruolo del microsporidio intestinale *Nosema ceranae* (nosemosi) nel collasso delle colonie [abstract]. Il filone, iniziato nel 2018 quando lo studente era ancora alle medie (articolo Inverse "Young Innovators": cinque alveari alimentati con barattoli contenenti *Bifidobacterium infantis* a concentrazioni diverse, due alte, due più basse, uno di controllo) [web: inverse.com], ha una domanda che evolve anno per anno:

- **Year 2 (ISEF 2019)**: un probiotico umano (*B. infantis*) riduce prevalenza/intensità di *Nosema* e migliora la "salute" dell'alveare? [dataset locale, ProjectId 17304]
- **Year 3 (ISEF 2020)**: in gabbia, *B. infantis* è efficace quanto la fumagillina (antifungino approvato FDA) e, a differenza di questa, migliora il microbiota del mesointestino? [ProjectId 19218]
- **Year 4 (ISEF 2021)**: si può definire l'indice terapeutico, cioè predire la % di riduzione di *Nosema* per ogni concentrazione iniziale e dose, con una regressione lineare multipla (Scikit-learn)? [ProjectId 20499]
- **Year 5 (ISEF 2022)**: dato che un modello batch (sklearn `LinearRegression`) "would not minimize the sum of the residuals unless the explicit training command was given", si può costruire un MLR "novel" con algoritmi di continual learning che si auto-aggiorna a ogni nuovo dato sperimentale, ottenendo errori più bassi delle librerie esistenti, e incapsularlo in un'app iOS per gli apicoltori? [abstract]

Il contributo "scientifico" (effetto del probiotico) è dunque consolidato negli anni 1-3; il contributo premiato nel 2022 è quasi interamente computazionale/traslazionale.

## Metodo passo per passo

Ricostruzione dalle quattro abstract ufficiali (2019-2022) e dall'articolo Inverse 2018; i passaggi marcati [inferenza] sono deduzioni dal testo, non dichiarazioni dell'autore.

1. **Trattamento probiotico (anni 1-3).** Miscela di batteri lattici somministrata nello sciroppo zuccherino: inizialmente solo *B. infantis* (2018-2020), poi miscela *B. infantis* + *B. bifidum* (scritto "B. bifidium" nell'abstract) + *Lactobacillus kunkeei* (un simbionte nativo dell'intestino delle api) [abstract 2022]. Dose documentata nel 2019: 500.000.000 CFU per alveare mescolati in soluzione zuccherina [ProjectId 17304].
2. **Studio di campo (Year 2).** 8 alveari: 4 trattati, 4 controlli con sciroppo semplice; osservazioni ogni due settimane per 6 settimane nei mesi invernali; variabili: conta di *Nosema*, peso del miele, popolazione di api e di covata [ProjectId 17304].
3. **Studio in gabbia controllato (Year 3).** Confronto a tre bracci [inferenza]: zucchero / fumagillina / *B. infantis*; conta di *Nosema* nel mesointestino di api infette e conta di unità formanti colonia (CFU) di batteri intestinali "favorevoli" con metodi colturali [ProjectId 19218].
4. **Conteggio delle spore.** Non descritto; il metodo standard [inferenza] è la macerazione degli addomi (tipicamente 10-30 api), diluizione e conta su emocitometro al microscopio ottico (spore/ape); la specie non è distinguibile al microscopio (vedi punti deboli).
5. **Costruzione del dataset per il modello (Year 4).** "Hundreds of trials from previous research and similar experiments" [ProjectId 20499]: dati propri degli anni 1-3 più prove di letteratura; variabili d'ingresso [inferenza]: concentrazione iniziale di *Nosema*, dose del probiotico (CFU), probabilmente tempo/giorni dal trattamento e composizione della miscela; uscita: % di riduzione (2021) o concentrazione predetta di *Nosema* (2022).
6. **MLR batch (Year 4).** Modello `sklearn.linear_model.LinearRegression` (minimi quadrati ordinari): y = β0 + β1x1 + … + βkxk + ε, con β = (XᵀX)⁻¹Xᵀy. Le predizioni venivano "periodically compared to experimental cage trials at various dosages" [ProjectId 20499].
7. **MLR con continual learning (Year 5).** Modello proprio che aggiorna i coefficienti a ogni nuovo punto sperimentale senza richiamare esplicitamente `fit()` sull'intero dataset [abstract]. L'abstract non specifica l'algoritmo; le due famiglie compatibili con "minimizzare continuamente la somma dei residui" sono [inferenza]: (a) minimi quadrati ricorsivi (RLS): K = P x/(λ + xᵀP x); β ← β + K (y − xᵀβ); P ← (P − K xᵀP)/λ; oppure (b) discesa del gradiente stocastica per campione: β ← β − η (xᵀβ − y) x.
8. **Validazione.** Confronto a ogni livello di dose tra valore predetto e valore sperimentale (margine d'errore), e confronto con "existing ML alternatives" (librerie esistenti) su accuratezza e tempo di calcolo [abstract].
9. **Traslazione.** Integrazione del modello in un'app iOS in cui l'apicoltore inserisce la conta di *Nosema* diagnosticata e ottiene la dose ideale di probiotico; nel 2022 l'integrazione era ancora in corso ("After continuing to integrate…") [abstract].

## Tecniche, strumenti, software e codice

- **Microbiologia applicata**: probiotici lattici (*Bifidobacterium infantis*, *B. bifidum*, *L. kunkeei*) in sciroppo; conte CFU su piastra per il microbiota del mesointestino (metodo colturale, non sequenziamento) [ProjectId 19218].
- **Parassitologia**: conte di spore di *Nosema ceranae* [abstract]; strumento non dichiarato (emocitometro + microscopio ottico è lo standard di settore) [inferenza].
- **Apicoltura**: alveari di campo (8 nel 2019, 5 nel 2018), gabbie sperimentali per api adulte (2020-2022) [ProjectId 17304, 19218; web: inverse.com].
- **Statistica**: t di Student a due campioni (2019) [ProjectId 17304]; significatività "extremely significant" per il miglioramento del modello (2021-2022) [abstract].
- **Software**: Python + Scikit-learn (`LinearRegression`) nel 2021 [abstract]; nel 2022 un modello MLR scritto in proprio con aggiornamento continuo, confrontato con "existing machine learning libraries" [abstract]; app iOS (quindi Swift/SwiftUI o Xcode + export del modello, p.es. Core ML) [inferenza dal termine "iOS application"].
- **Codice**: nessun codice pubblico trovato. Ricerche eseguite: GitHub API/MCP per utenti "Varun Madan" (4 account; `madanva` è l'autore, bio "cs @ stanford | hoping to build cool stuff that benefits society", organizzazione GDSC-Stanford, 24 repo tutte successive al giugno 2023: giochi JavaScript, mappe R/HTML, progetti Stanford CS131/CS217/CS244C/CS336/AA228, nessuna su api); `varunmadan1014` vuoto; code search GitHub per "nosema" + "linear regression", "kunkeei infantis regression", "nosema dosage language:python", "nosema extension:ipynb", "nosema language:swift": nessun risultato riconducibile al progetto [web: github.com/madanva; api.github.com]. 
- **Cosa doveva contenere il software (ricostruzione)**: (1) un loader del dataset (CSV con colonne tipo `initial_nosema`, `dose_cfu`, `days`, `final_nosema`); (2) una classe `ContinualMLR` con metodi `predict(x)` e `update(x, y)` che mantiene β e la matrice P (RLS) o esegue un passo di SGD; (3) uno script di confronto con `sklearn.linear_model.LinearRegression`/`SGDRegressor` e misura dei tempi (`time.perf_counter`); (4) calcolo dell'errore per dose e della "accuracy" (verosimilmente 100 − errore percentuale assoluto medio); (5) grafici predetto-vs-osservato; (6) un'app iOS che inverte il modello: data la conta iniziale e la riduzione target, restituisce la dose.

## Dati, campioni e statistica

- 2018: 5 alveari (2 dose alta, 2 dose bassa, 1 controllo) [web: inverse.com].
- 2019: 8 alveari (4 trattati con 5×10⁸ CFU, 4 controlli), 6 settimane, rilievi bisettimanali (quindi ~3-4 rilievi per alveare), inverno; t-test a due campioni, p < 0,05 su conta *Nosema*, peso del miele, popolazione di api e covata [ProjectId 17304].
- 2020: studio in gabbia (numero di gabbie e di api per gabbia: non trovato); conte di *Nosema* e CFU del microbiota [ProjectId 19218].
- 2021: dataset di addestramento di "hundreds of trials" (numero esatto: non trovato), validazione con prove in gabbia a varie dosi [ProjectId 20499].
- 2022: confronto predetto/sperimentale "at every treatment dosage value" (numero di dosi: non trovato); differenza modello-sperimento "insignificant" dopo l'addestramento; accuratezza di picco 92,8% contro le librerie esistenti [abstract; web: projectboard.world]. Metriche precise (MAE, R², intervallo di confidenza), split train/test e cross-validation: non trovati.

## Risultati chiave (numeri)

- Api responsabili di "over 80 percent of worldwide pollination" [abstract] (dato di contesto citato dallo studente).
- Dose probiotica di campo: 500.000.000 CFU/alveare (2019) [ProjectId 17304].
- *B. infantis* riduce significativamente la conta di *Nosema* e aumenta peso del miele, api e covata, p < 0,05 (2019) [ProjectId 17304].
- *B. infantis* "as effective as Fumagillin" nel ridurre *Nosema* in gabbia, con in più un aumento delle CFU di batteri intestinali favorevoli (2020) [ProjectId 19218].
- Modello MLR 2021: "margin of error … significantly decreased at every treatment dosage value" [ProjectId 20499].
- Modello MLR continual 2022: accuratezza di picco **92,8%**, "significantly more accuracy" delle librerie ML esistenti "without sacrificing significant processing time"; differenza predetto-sperimentale non significativa [abstract; web: projectboard.world].
- Premi: $5.000 (First Award, Animal Sciences 2022; una delle due First Award della categoria su 75 righe di premio ANIM 2022 nel dataset locale) + EUCYS Award; $5.000 nel 2021 [web: societyforscience.org; dataset locale].

## Perché ha vinto — analisi secondo i criteri ISEF

Criteri Science (ANIM): Research Question 10%, Design & Methodology 15%, Execution 20%, Creativity 20%, Presentation 35%.

- **Research Question (10%)**: problema reale, quantificato e socialmente rilevante (impollinazione, nosemosi); la domanda del 2022 è specifica e falsificabile (il modello continuo batte il batch?). Punto forte: la domanda nasce da un limite concreto incontrato l'anno prima, il che mostra ai giudici un ricercatore che itera.
- **Design & Methodology (15%)**: progressione classica da campo (8 alveari, controlli, inverno) a gabbia (comparatore attivo: fumagillina, standard FDA) a modello predittivo validato contro nuovi esperimenti. Il confronto con un comparatore attivo e non solo con il placebo è ciò che distingue il progetto dalla media dei progetti ANIM.
- **Execution (20%)**: cinque anni di dati propri, test statistici dichiarati (t-test, p < 0,05), validazione del modello "a ogni dose". I giudici premiano la coerenza longitudinale: ogni anno chiude un'ipotesi e ne apre una nuova.
- **Creativity (20%)**: unire microbiologia dell'alveare e machine learning in categoria Animal Sciences (dove i modelli ML sono rari) e trasformare il risultato in uno strumento per l'utente finale (app per la dose). Il titolo "Year 5" segnala persistenza; la "novità" del MLR continuo è il gancio narrativo.
- **Presentation (35%)**: la stessa impostazione aveva già vinto un First Award nel 2021 e l'abstract 2022 è scritto come una storia (problema → limite tecnico → soluzione → numero → applicazione), il che è esattamente quanto conviene nell'intervista di 25 punti. Il premio EUCYS (selezione aggiuntiva da parte di Society for Science) conferma che la comunicazione era di livello.

Contesto: nel 2022 Animal Sciences ha assegnato due First Award ($5.000): Yi-Shan Hung (Taiwan, *Drosophila*) e Madan [dataset locale]; altri progetti sulle api dello stesso anno (vibroacustica con HMM, computer vision per la salute dell'alveare) hanno ottenuto solo Third Award o premi speciali, segno che a fare la differenza non è stato il tema ma la profondità pluriennale e la validazione sperimentale.

## Punti deboli e domande da giudice

1. **"Novel MLR"?** La regressione lineare con aggiornamento ricorsivo (RLS, anni '50) e `SGDRegressor.partial_fit` di Scikit-learn esistono già. Domanda: "In cosa il tuo algoritmo differisce da RLS o da `partial_fit`? Hai confrontato con `SGDRegressor`, o solo con `LinearRegression` batch?"
2. **Metrica "92,8%" non definita**: accuratezza di cosa? 100 − MAPE? Entro quale tolleranza? Domanda: "Riporta MAE, RMSE e R² su un test set tenuto fuori; qual è l'intervallo di confidenza?"
3. **Rischio di leakage**: se il modello si aggiorna con i dati sperimentali e poi viene valutato sugli stessi, l'"insignificant difference" è attesa. Domanda: "La validazione è stata fatta prima o dopo l'aggiornamento con quel punto?"
4. **Linearità**: la riduzione di *Nosema* in funzione della dose è plausibilmente saturante (log-dose); perché un modello lineare e non log-lineare o logistico?
5. **Numerosità**: 4 vs 4 alveari (2019) con rilievi ripetuti pone un problema di pseudo-replicazione; il t-test non tiene conto delle misure ripetute. Numero di gabbie/api per braccio: non dichiarato.
6. **Identificazione della specie**: la conta su emocitometro non distingue *N. ceranae* da *N. apis*; serve PCR.
7. **Microbiota solo colturale**: senza 16S rRNA, "miglioramento del microbiota" è una misura parziale (molti simbionti delle api non crescono in coltura standard).
8. **Probiotici umani**: *B. infantis*/*B. bifidum* non sono simbionti nativi delle api; colonizzano o agiscono solo transitoriamente? Effetti a lungo termine sulla colonia?
9. **Confondenti stagionali** (inverno) e assenza di cecità nei rilievi.
10. **App iOS non completata** al momento della fiera: "quale parte del prodotto funziona davvero oggi?"
11. **Riproducibilità**: nessun codice, dataset o protocollo pubblico.

## Percorso dello studente dopo ISEF

- Settembre 2022: rappresentante USA a EUCYS 2022 (Leiden, Paesi Bassi) con lo stesso progetto [web: eucysleiden2022.eu; societyforscience.org]; eventuale premio EUCYS: non trovato.
- Università: Stanford University, School of Engineering, informatica ("cs @ stanford") [web: linkedin.com/in/varun-madan-8817231b1; github.com/madanva]. Le repository pubbliche 2023-2026 mostrano corsi Stanford di computer vision (CS131), modelli linguistici (CS336), hardware (CS217), reti (CS244C), decision making (AA228/CS238) e progetti personali (mappe di rifugi per senzatetto a San Francisco, mappe di centri di riciclo, strumenti AI) [web: github.com/madanva]; nessuna continuazione pubblica del lavoro sulle api.
- Startup, brevetti, pubblicazioni successive sul tema: non trovati.

## Lezioni per un progetto EAEV/PHYS su radon e bradisismo

1. **Costruire un arco pluriennale esplicito.** Il titolo "Year 5" e gli abstract che richiamano gli anni precedenti hanno fatto percepire profondità. Per il radon: Anno 1 = serie temporali e modello fisico di trasporto (diffusione-avvezione di Rn-222/Rn-220 nel suolo), Anno 2 = validazione in campo ai Campi Flegrei/Ischia, Anno 3 = strumento operativo per l'Osservatorio Vesuviano o la Protezione Civile.
2. **Il limite tecnico di un anno diventa la domanda dell'anno dopo.** Madan ha trasformato un fastidio pratico (il modello batch non si aggiorna) nel cuore del progetto. Analogia diretta: un modello di trasporto del radon con parametri fissi (coefficiente di diffusione, velocità di avvezione, emanazione) non segue le variazioni di permeabilità e flusso di gas durante l'unrest; proporre un'assimilazione dati sequenziale (RLS/filtro di Kalman) che aggiorna i parametri a ogni nuova misura è l'equivalente, molto più fisico, del suo "continual learning".
3. **Confronto con un comparatore forte, non con il nulla.** Lui ha confrontato con fumagillina (standard) e con Scikit-learn. Per il radon: confrontare il proprio modello con un modello standard di letteratura (p.es. soluzione analitica di diffusione stazionaria, o regressione radon-vs-pressione/temperatura) e con le anomalie sismiche/deformative INGV.
4. **Definire le metriche prima di presentare.** Il "92,8%" è attaccabile; usare MAE/RMSE/R², set di test separato, intervalli di confidenza, e dichiarare la separazione tra dati di calibrazione e di validazione.
5. **Prodotto per l'utente finale.** L'app iOS ha pesato sulla creatività e sulla presentazione; per il radon, una dashboard (anche web) che riceve la serie di radon e restituisce il parametro di trasporto stimato e un indice di anomalia è l'analogo credibile. Ma deve funzionare in fiera, non "in integrazione".
6. **Riprodurre e documentare.** L'assenza di codice e dataset pubblici è un punto debole evitabile: pubblicare repository, dati grezzi, protocollo di misura (strumento radon, calibrazione, posizione sonde, profondità, umidità, pressione).
7. **Anticipare le domande sulle alternative.** Preparare una risposta precisa a "perché non il metodo X già esistente?" (per il radon: perché non un semplice modello statistico di correlazione con parametri meteorologici?), mostrando dove il modello fisico spiega ciò che la statistica non spiega.
8. **Categoria e giuria.** Un progetto computazionale in una categoria sperimentale (ANIM) si è distinto per contrasto; in EAEV/PHYS, dove i modelli sono attesi, la differenza la faranno la validazione sul campo e la fisica del processo (sorgente, emanazione, diffusione, avvezione, decadimento di Rn-220 vs Rn-222 come traccianti di profondità).

## Fonti

- Record locale del dataset ISEF (abstract ufficiale 2022, ProjectId 21991): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=&Category=&AllAbstracts=True&FairCountry=&FairState=&ProjectId=21991
- Abstract ISEF 2021 "Year 4" (ProjectId 20499): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=&Category=&AllAbstracts=True&FairCountry=&FairState=&ProjectId=20499
- Abstract ISEF 2020 "Year Three" (ProjectId 19218): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=&Category=&AllAbstracts=True&FairCountry=&FairState=&ProjectId=19218
- Abstract ISEF 2019 "Year Two" (ProjectId 17304): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=&Category=&AllAbstracts=True&FairCountry=&FairState=&ProjectId=17304
- ProjectBoard ISEF 2022, ANIM011: https://projectboard.world/isef/project/anim011---optimizing-bee-gut-immunity-with-a-novel-mlr-model
- ISEF 2021 virtuale, ANIM007: https://isef.net/project/anim007---optimizing-bee-gut-immunity-with-a-novel-mlr-model
- EUCYS Leiden 2022, pagina del progetto: https://eucysleiden2022.eu/year-5-developing-a-novel-multiple-linear-regression-model-to-optimize-honey-bee-gut-immunity-using-a-lactic-acid-bacteria-probiotic-mixture/
- Society for Science, Full Awards ISEF 2022: https://www.societyforscience.org/press-release/regeneron-isef-full-awards-2022/
- Society for Science, EU Contest for Young Scientists Award: https://www.societyforscience.org/isef/awards/eu-young-scientists-award/
- Society for Science, Full Awards ISEF 2021: https://www.societyforscience.org/press-release/2021-regeneron-isef-grand-awards/
- Regeneron ISEF 2022 Finalist Directory (PDF, non apribile dalla rete di lavoro): https://sspcdn.blob.core.windows.net/files/Documents/SEP/ISEF/2022/Finalist-Directory.pdf
- Inverse, "Orlando Teen Varun Madan Uses Probiotic Bacteria to Save the Honeybees" (2018): https://www.inverse.com/article/50242-can-bacteria-save-the-bees-young-innovators
- LinkedIn, Varun Madan — Stanford University School of Engineering: https://www.linkedin.com/in/varun-madan-8817231b1/
- GitHub, profilo dell'autore (nessun codice del progetto): https://github.com/madanva
- GitHub API, ricerca utenti e repository (nessun risultato pertinente): https://api.github.com/search/users?q=Varun+Madan ; https://api.github.com/search/repositories?q=honey+bee+nosema+regression
- Lista premi ISEF 2022 (aggregatore, non apribile): https://www.hanlin.com/archives/586729
- Letteratura di contesto sui probiotici anti-Nosema (per valutare cosa esisteva già): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7996622/ ; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8226692/ ; https://doi.org/10.3390/pathogens11111269

Nota metodologica: in questa sessione il budget di WebSearch era quasi esaurito (4 ricerche eseguite) e tutti i siti non-GitHub (Society for Science, ProjectBoard, EUCYS, Inverse, LinkedIn, archivi web, Crossref, PubMed) erano bloccati dal proxy; i contenuti di quelle pagine sono stati ottenuti solo attraverso i riassunti del motore di ricerca.
