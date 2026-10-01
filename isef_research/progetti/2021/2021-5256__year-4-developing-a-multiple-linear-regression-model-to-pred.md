# Year 4: Developing a Multiple Linear Regression Model to Predict the Specific Effects of Various Lactic Acid Bacteria Dosages on the Overall Honey Bee Gut Microbiota and Nosema ceranae Reduction (2021)

> **Nota sulle fonti.** Questa scheda è stata compilata con 4 ricerche WebSearch andate a buon fine (il budget di ricerca della sessione si è esaurito subito dopo: 10 ulteriori query sono state rifiutate) e con il dataset locale degli abstract ISEF, che contiene **tutti e quattro gli abstract ufficiali** della serie (ISEF 2019, 2020, 2021, 2022). Quasi tutti i siti esterni (Society for Science, Bee Culture, Inverse, Stanford, EUCYS, OSC, PDF delle directory) sono bloccati in WebFetch: i loro contenuti sono stati letti **solo attraverso i riassunti restituiti da WebSearch** e sono marcati `[web: URL]`. GitHub è stato consultato direttamente. Ciò che viene solo dall'abstract 2021 è marcato `[abstract]`; ciò che viene dagli abstract degli altri anni è marcato `[abstract 2019]`, `[abstract 2020]`, `[abstract 2022]`.

## Scheda

| Campo | Valore |
|---|---|
| Anno | 2021 (Regeneron ISEF 2021, edizione interamente virtuale) |
| Premi (tutti, con importi) | **First Award — $5,000** (Grand Award, 1° posto di categoria Animal Sciences) [record ISEF; web: societyforscience.org/press-release/2021-regeneron-isef-grand-awards/]; **University of Arizona: Renewal Tuition Scholarship** (Special Award, borsa di studio rinnovabile, importo non trovato) [record ISEF] |
| Categoria | Animal Sciences (ANIM) |
| Finalista/i | Varun Madan (progetto individuale) |
| Scuola | Lake Highland Preparatory School |
| Luogo | Orlando, Florida (FL), Stati Uniti [record ISEF; web: Broadcom MASTERS 2018] |
| Mentore / laboratorio | Nome del mentore: **non trovato**. Contesto: programma di ricerca scientifica della Lake Highland Preparatory School + collaborazione con la **University of Florida** [web: engineering.stanford.edu/spotlight/varun-madan]. L'idea nacque da una conferenza di un professore universitario sui problemi immunitari delle api [web: Broadcom MASTERS 2018] |
| Codice pubblico | **nessun codice pubblico trovato** (profilo GitHub dell'autore identificato, `github.com/madanva`, 24 repository, nessuno sul progetto; vedi sezione dedicata) |
| Pubblicazioni | Nessun articolo peer-reviewed trovato. Esposizione pubblica: pagina progetto EUCYS Leiden 2022 (Year 5); articolo "Optimizing Bee Gut Immunity" su *Bee Culture*; articolo *Inverse* (2018) |
| Tipo di ricerca | Serie pluriennale (5 anni) di ricerca sperimentale applicata (apicoltura / microbiologia) che nell'anno 4 diventa **modellazione statistica/ML** (regressione lineare multipla) validata contro prove sperimentali in gabbia. Categoria *Science* |

## Abstract ufficiale (EN)

Testo integrale dal record ISEF 2021 (ProjectId 20499) [abstract]:

> "Honeybees are essential to society, performing over 80 percent of worldwide pollination. However, commercial hive populations have been decreasing at an alarming rate in the United States. Multiple studies have shown that deteriorating gut health is one of the major reasons for their maladaptive response to various external stressors. Research throughout the last three years revealed that treating the hives with a Bifidobacterium infantis probiotic significantly reduced the counts of the harmful gut parasite Nosema ceranae. In addition, the treatment significantly improved honeybee midgut bacterium counts and overall hive health. In order to extrapolate these monumental findings to widespread farming practices, the current project aims to configure a method to better define the therapeutic index of the previously proven bacterial treatment. To this end, a multiple linear regression model was developed to predict the percentage reduction in the Nosema counts for any given initial concentration and treatment dosage. After training this model with hundreds of trials from previous research and similar experiments, the output values were periodically compared to experimental cage trials at various dosages. The results showed an extremely significant improvement in output accuracy, as the margin of error in predicted Nosema concentration significantly decreased at every treatment dosage value throughout the experiment. After continuing this model at an even larger scale, farmers will potentially determine the ideal dosage of the bacterial treatment moments after recording their Nosema counts. This probiotic treatment technology has the potential to improve hive immunity and agricultural productivity throughout the world."

## Problema e domanda di ricerca

**Contesto.** Le api mellifere svolgono "over 80 percent of worldwide pollination" [abstract]; negli USA gli alveari gestiti sono passati da circa 6 milioni di 70 anni fa a meno di 3 milioni [web: societyforscience.org/broadcom-masters/broadcom-masters-2018-finalists/]. Il parassita microsporidio *Nosema ceranae* infetta l'intestino medio dell'ape ed è associato al collasso delle colonie; l'unico trattamento storico, la **fumagillina**, è un antifungino che "causes midgut epithelial degeneration" [abstract 2019].

**Cosa esisteva prima dell'anno 4 (lavoro dello stesso studente).**
- *Anno 1 (2017-18, 8ª classe, Broadcom MASTERS 2018)*: "Field Testing of Feeding Bacterium Bifidobacterium infantis (found in a human gut probiotic) in Order to Improve Honey Bee Health" — primo test di campo del probiotico umano *B. infantis* sulle api [web: Broadcom MASTERS 2018].
- *Anno 2 (ISEF 2019, "Year Two: Understanding the Effects of Bifidobacterium infantis on Honeybee Gut Parasite Nosema ceranae")*: 8 alveari, 4 trattati con 5×10⁸ CFU in soluzione zuccherina e 4 controlli; osservazioni bisettimanali per 6 settimane in inverno; t di Student a due campioni; riduzione significativa del conteggio di Nosema e aumento di peso del miele, popolazione di api e covata con p < 0,05 [abstract 2019].
- *Anno 3 (ISEF 2020, "Year Three: Evaluating the Effects of Bifidobacterium infantis Compared with Fumagillin...")*: studio controllato in gabbia, *B. infantis* vs fumagillina vs zucchero; microbiota del midgut misurato con metodi colturali (CFU); il probiotico risultò "as effective as Fumagillin in decreasing Nosema counts" e aumentò le CFU di batteri favorevoli [abstract 2020].

**Domanda dell'anno 4.** Il trattamento funziona, ma qual è la **dose giusta** per un dato livello di infezione? L'obiettivo dichiarato è "configure a method to better define the therapeutic index of the previously proven bacterial treatment" [abstract], cioè costruire un modello predittivo che, dati (1) la concentrazione iniziale di Nosema e (2) la dose di batteri lattici, restituisca la **riduzione percentuale attesa** del conteggio di Nosema, così che un apicoltore possa scegliere la dose "moments after recording their Nosema counts" [abstract].

## Metodo passo per passo

Ricostruzione dall'abstract 2021, integrata con quanto dichiarato negli abstract 2020 e 2022 (che descrivono retrospettivamente l'anno 4).

1. **Costruzione del dataset di addestramento.** Raccolta di "hundreds of trials from previous research and similar experiments" [abstract]: i dati di campo e di gabbia degli anni 1-3 (conteggi di Nosema pre/post trattamento a dose nota) più dati di esperimenti analoghi in letteratura. Ogni record = (concentrazione iniziale di Nosema, dose di probiotico) → riduzione percentuale di Nosema. Il numero esatto di record e le fonti esterne usate: non trovati.
2. **Scelta del modello.** Regressione lineare multipla (MLR) con due predittori: il modello è stato implementato con **Scikit-learn** ("An existing Scikit-learn multiple linear regression model was used last year") [abstract 2022]. Forma: `riduzione% = β₀ + β₁·(Nosema iniziale) + β₂·(dose CFU)` (eventuali termini di interazione o trasformazioni logaritmiche: non trovati).
3. **Addestramento** sul dataset storico (stima OLS dei coefficienti; in Scikit-learn `LinearRegression.fit` risolve i minimi quadrati via decomposizione SVD/lstsq).
4. **Prove sperimentali in gabbia a dosi diverse.** Api in gabbia infettate con *N. ceranae* e alimentate con soluzione zuccherina contenente dosi variabili di batteri lattici (replicando il protocollo dell'anno 3 [abstract 2020]); conteggio di Nosema a fine prova. Le dosi testate e il numero di gabbie: non trovati.
5. **Confronto periodico modello-esperimento.** "the output values were periodically compared to experimental cage trials at various dosages" [abstract]: per ogni dose si confronta la riduzione predetta con quella misurata e si calcola il margine di errore.
6. **Ri-addestramento iterativo.** Ogni nuovo blocco di risultati sperimentali viene aggiunto al training set e il modello riaddestrato; l'esito riportato è che "the margin of error in predicted Nosema concentration significantly decreased at every treatment dosage value throughout the experiment" [abstract]. (È proprio il fatto che Scikit-learn non riaggiorna i coefficienti senza una chiamata esplicita di `fit` il "problema" che motiva l'anno 5: "the model would not minimize the sum of the residuals unless the explicit training command was given" [abstract 2022].)
7. **Uso previsto.** Lettura del conteggio di Nosema → inserimento nel modello → dose consigliata; nell'anno 5 questo flusso diventa un'app iOS [abstract 2022].

## Tecniche, strumenti, software e codice

**Tecniche biologiche (ereditate dagli anni 1-3).**
- Alimentazione di alveari/gabbie con probiotico in soluzione zuccherina (dose di riferimento 5×10⁸ CFU) [abstract 2019].
- Conteggio delle spore di *Nosema ceranae* nel midgut ("Nosema counts"); il metodo di conteggio (tipicamente omogenato addominale + emocitometro) **non è specificato negli abstract**.
- Conta colturale (CFU) dei batteri del midgut per valutare il microbiota [abstract 2020].
- Confronto con fumagillina come controllo positivo [abstract 2020].
- Nell'anno 5 il probiotico diventa una miscela *B. infantis* + *B. bifidum* (scritto "B. bifidium") + *Lactobacillus kunkeei* [abstract 2022]; nel titolo dell'anno 4 compare già "Various Lactic Acid Bacteria Dosages", quindi è probabile che la miscela fosse in uso nel 2021, ma l'abstract 2021 cita esplicitamente solo *B. infantis*.

**Software.**
- Linguaggio: **Python** (dedotto dall'uso dichiarato di Scikit-learn) [abstract 2022]. Modello: `sklearn.linear_model.LinearRegression` (OLS) [abstract 2022]; nel 2022 lo studente confronta la propria MLR "continual-learning" con "existing machine learning libraries" raggiungendo "peak of 92.8%" di accuratezza [abstract 2022].
- Statistica degli anni precedenti: t di Student a due campioni [abstract 2019].

**Codice pubblico: nessun codice pubblico trovato.**
- Ricerca utenti GitHub "varun madan" → 4 account; `github.com/madanva` è l'autore (bio "cs @ stanford", 24 repository, organizzazione GDSC-Stanford) [web: https://github.com/madanva]. I repository sono tutti universitari o personali (fork di Stanford CS336, CS217 kernel fusion in Verilog, CS244C, AA228 in Julia, `priori-ai`, `haas-ai-project`, `shelter-map`, `interactive-recycling-maps` in R/Shiny/Leaflet, `Tenzies`, `JavaScript-Casino`); nessuno riguarda api, Nosema, probiotici o regressione. `github.com/varunmadan1014` ha 0 repository.
- Ricerca repository GitHub: "Nosema honey bee" → 3 repo di terzi non collegati (jscscheper/within-hive-dynamics, modello matematico in R; sydmil/PLOSOne-Innate-Defenses; Pcariman/16S-Assignments-DADA2); "Nosema ceranae regression", "honey bee probiotic", "Nosema dosage", "beekeeper Nosema app", "honeybee probiotic dosage regression" → 0 risultati.

**Cosa doveva contenere il software (ricostruzione dall'abstract).** Un notebook/script Python con: (a) caricamento di un CSV con colonne `nosema_iniziale`, `dose_CFU`, `riduzione_pct`; (b) `train_test_split` o validazione sequenziale per blocchi; (c) `LinearRegression().fit(X, y)`; (d) `predict` sulle dosi delle prove in gabbia; (e) calcolo dell'errore (differenza percentuale o MAE) per dose e per iterazione; (f) grafici errore-vs-iterazione. L'anno 5 aggiunge un aggiornamento online dei coefficienti (equivalente concettuale di `SGDRegressor.partial_fit`) e un'app iOS (Swift) che incapsula il modello [abstract 2022].

## Dati, campioni e statistica

- **Training set**: "hundreds of trials" [abstract] — numero esatto, provenienza (quota propria vs letteratura) e intervalli di dose/concentrazione: non trovati.
- **Validazione**: prove in gabbia "at various dosages" ripetute "periodically" [abstract]; numero di gabbie, api per gabbia, numero di dosi, durata: non trovati. Protocollo plausibile = quello dell'anno 3 (gabbie con api infettate, confronto fra trattamenti) [abstract 2020].
- **Dati storici propri**: anno 2 = 8 alveari (4+4), 6 settimane, osservazioni bisettimanali, dose 5×10⁸ CFU [abstract 2019]; anno 3 = studio in gabbia a tre bracci (probiotico, fumagillina, zucchero) [abstract 2020].
- **Statistica**: la metrica dichiarata è il "margin of error in predicted Nosema concentration" per ciascuna dose; il miglioramento è definito "extremely significant" [abstract] ma il test usato e i valori di p non compaiono nell'abstract. Per gli anni 2-3 il test è il t di Student a due campioni, p < 0,05 [abstract 2019]. R², RMSE, intervalli di confidenza dei coefficienti: non trovati.

## Risultati chiave (numeri)

- 1° posto Animal Sciences, **$5,000** + University of Arizona Renewal Tuition Scholarship [record ISEF].
- Modello MLR a 2 ingressi → riduzione % di Nosema; errore di previsione "significantly decreased at every treatment dosage value" durante l'esperimento [abstract].
- Numeri ereditati: dose 5×10⁸ CFU; 8 alveari; 6 settimane; p < 0,05 su Nosema, peso del miele, popolazione e covata [abstract 2019]; probiotico statisticamente equivalente alla fumagillina nel ridurre Nosema, con CFU di batteri favorevoli aumentate [abstract 2020; web: beeculture.com/optimizing-bee-gut-immunity/].
- Anno 5 (continuazione): accuratezza di picco **92,8 %** della MLR "continual-learning" contro le librerie ML esistenti, "without sacrificing significant processing time" [abstract 2022]; First Award $5,000 + **EU Contest for Young Scientists Award** a ISEF 2022, con partecipazione a EUCYS Leiden 2022 [record ISEF 2022; web: eucysleiden2022.eu; web: societyforscience.org/blog/experiencing-stem-and-stroopwafels-at-2022-eucys/].

## Perché ha vinto — analisi secondo i criteri ISEF

Criteri *Science*: Research Question 10 %, Design & Methodology 15 %, Execution 20 %, Creativity 20 %, Presentation 35 % (poster 10 % + colloquio 25 %).

- **Research Question (10 %)**: domanda concreta e "a valle" di tre anni di risultati positivi: non "funziona?" ma "quanto ne serve?". Un giudice vede subito il salto da efficacia a **indice terapeutico**, cioè verso l'applicazione reale.
- **Design & Methodology (15 %)**: ciclo modello → esperimento → ri-addestramento, con validazione contro dati nuovi e non solo contro il training set. È un disegno semplice ma corretto per un problema di dosaggio, e usa un modello interpretabile (coefficienti leggibili come "effetto per unità di dose").
- **Execution (20 %)**: il punto forte è la **continuità sperimentale**: lo studente aveva alveari in campo, gabbie infettate, conteggi di Nosema e conte CFU già rodati [abstract 2019, 2020]; i dati di addestramento erano in gran parte suoi. In categoria Animal Sciences, dove molti progetti sono osservazionali, avere un sistema sperimentale controllato per quattro anni è raro.
- **Creativity (20 %)**: trasferire un probiotico umano (*B. infantis*) alle api era già un'idea originale nel 2018 [web: inverse.com]; nel 2021 l'originalità sta nel trattare il dosaggio come problema di regressione e nel pensare al prodotto finale (apicoltore → conteggio → dose).
- **Presentation (35 %)**: ISEF 2021 era virtuale, quindi il colloquio pesava molto. Una narrazione a cinque anni ("Year 4" nel titolo è una scelta deliberata) permette di rispondere a qualsiasi domanda sul sistema biologico con dati propri; l'abstract usa la struttura classica problema → risultati precedenti → obiettivo → metodo → risultato → impatto.

Un fattore non tecnico ma reale: la **traiettoria** (Broadcom MASTERS 2018 finalist [web], poi ISEF 2019, 2020, 2021, 2022) segnala ai giudici maturità e padronanza del tema.

## Punti deboli e domande da giudice

1. **"Hundreds of trials" da dove?** Se gran parte dei dati proviene da esperimenti di altri (ceppi, dosi e metodi di conteggio diversi), il modello mescola popolazioni eterogenee: come è stata normalizzata la dose (CFU/ape? CFU/alveare?) e la concentrazione (spore/ape)?
2. **Linearità**: la risposta dose-effetto di un probiotico è tipicamente saturante (sigmoide o log-dose). Perché una MLR lineare e non una regressione su log(dose) o un modello Emax? Sono stati verificati i residui?
3. **Metriche**: "margin of error" non è definito; mancano R², RMSE, intervalli di confidenza e il test che giustifica "extremely significant".
4. **Rischio di circolarità**: se le prove in gabbia usate per il confronto vengono poi aggiunte al training, l'errore scende per costruzione; serviva un hold-out mai usato in addestramento.
5. **Dimensione campionaria delle gabbie**: quante gabbie per dose, quante api, quante repliche? Il controllo con sola soluzione zuccherina era presente anche nell'anno 4?
6. **Ceppi**: il titolo parla di "various lactic acid bacteria", l'abstract solo di *B. infantis*; quando è entrata la miscela con *B. bifidum* e *L. kunkeei* e il modello ne tiene conto (variabile categoriale)?
7. **Vitalità del probiotico** nello sciroppo (CFU effettive al momento dell'assunzione) e identificazione di specie di *Nosema* (ceranae vs apis) con PCR: non trovati.
8. **Il "problema" di Scikit-learn** dichiarato nel 2022 ("would not minimize the sum of the residuals unless the explicit training command was given") è in realtà il normale comportamento di qualsiasi stimatore batch; un giudice informatico chiederebbe perché non usare semplicemente `partial_fit`/SGD o un filtro ricorsivo, e che cosa significhi "accuracy 92.8 %" per una regressione.

## Percorso dello studente dopo ISEF

- 2022: ISEF 2022, "Year 5", First Award $5,000 + EU Contest for Young Scientists Award [record ISEF 2022]; rappresentante USA a EUCYS Leiden 2022 con lo stesso progetto [web: eucysleiden2022.eu; web: societyforscience.org/blog/experiencing-stem-and-stroopwafels-at-2022-eucys/]. Eventuale premio EUCYS: non trovato.
- Università: Computer Science alla **Stanford University** (profilo ufficiale "Spotlight: Varun Madan" della Stanford School of Engineering, che riassume il progetto come modello ML per la dose ottimale di un probiotico sviluppato "through his high school's research program and the University of Florida") [web: engineering.stanford.edu/spotlight/varun-madan]. Il profilo GitHub pubblico mostra interessi in AI/ML (corsi Stanford CS336, CS217, CS244C, AA228, CS131) e progetti civici (`shelter-map`, `interactive-recycling-maps`) [web: github.com/madanva].
- Altre partecipazioni citate: Dr. Nelson Ying Science Competition (Orlando Science Center) [web: osc.org] — anno e premio non trovati; Broadcom MASTERS 2018 finalist [web: societyforscience.org/broadcom-masters/broadcom-masters-2018-finalists/]. Startup, brevetti, pubblicazioni successive: non trovati.

## Lezioni per un progetto EAEV/PHYS su radon e bradisismo

1. **Il titolo "Year N" come strategia**: dichiarare la continuità pluriennale trasforma il progetto in un programma di ricerca. Per il radon: "Year 1: misure e modello 1-D di diffusione/avvezione" → "Year 2: calibrazione sui dati INGV di Campi Flegrei/Ischia" → "Year 3: previsione".
2. **Dal "funziona" al "quanto"**: il salto premiato nel 2021 è passare da un effetto qualitativo a una **relazione quantitativa utilizzabile** (dose ↔ riduzione). Analogo radon: non limitarsi a "il radon aumenta prima della crisi", ma stimare quanto varia il flusso di Rn-222/Rn-220 per una data variazione di permeabilità, gradiente di pressione o temperatura, con un modello fisico (equazione di diffusione-avvezione con sorgente e decadimento) e coefficienti stimati per regressione.
3. **Validazione contro dati nuovi, non contro il training**: il punto più fragile del progetto (rischio di circolarità) è esattamente ciò che un modello radon deve evitare: calibra sul 2019-2022 e verifica sul 2023-2024, oppure su un sito diverso (Ischia vs Solfatara).
4. **Scegliere la forma funzionale giusta**: una MLR lineare su un fenomeno saturante è attaccabile; per il radon, la fisica suggerisce dipendenze esponenziali (profondità, lunghezza di diffusione √(D/λ)) e il modello deve rifletterle. Mostrare i residui.
5. **Definire le metriche** (RMSE, R², intervalli di confidenza, test statistico) e scriverle nell'abstract: l'assenza di numeri è il principale difetto di questo abstract vincente; un progetto PHYS/EAEV non può permettersela.
6. **Pensare all'utente finale**: "the farmer records the Nosema count and gets the dose" è una frase che i giudici ricordano. Analogo: "la Protezione Civile legge il flusso di radon e il modello restituisce l'anomalia attesa rispetto alle variazioni meteorologiche", magari in una semplice dashboard.
7. **Possedere il sistema sperimentale**: quattro anni di alveari propri hanno reso lo studente inattaccabile sul contesto biologico. Per il radon, anche una piccola stazione propria (rivelatore a tracce o a camera ionizzante, sensori di pressione/temperatura/umidità del suolo) accanto ai dati pubblici INGV vale più di soli dati scaricati.
8. **Codice come prova**: non è stato trovato codice pubblico; pubblicare un repository documentato (dati, notebook di calibrazione, script del modello) è un vantaggio che i vincitori del 2021 non avevano e che oggi i giudici apprezzano.

## Fonti

- Record ISEF 2021, abstract ufficiale (ProjectId 20499): https://abstracts.societyforscience.org/Home/FullAbstract?Category=Any+Category&AllAbstracts=True&FairCountry=Any+Country&FairState=Any+State&ProjectId=20499
- Record ISEF 2019, "Year Two..." (ProjectId 17304): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=0%2C&Category=Any%20Category&AllAbstracts=True&FairCountry=Any%20Country&FairState=Any%20State&ProjectId=17304
- Record ISEF 2020, "Year Three..." (ProjectId 19218): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=0%2C&Category=Any%20Category&AllAbstracts=True&FairCountry=Any%20Country&FairState=Any%20State&ProjectId=19218
- Record ISEF 2022, "Year 5..." (ProjectId 21991): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=0%2C&Category=Any%20Category&AllAbstracts=True&FairCountry=Any%20Country&FairState=Any%20State&ProjectId=21991
- Society for Science, premi ISEF 2021: https://www.societyforscience.org/press-release/2021-regeneron-isef-grand-awards/
- Society for Science, premi ISEF 2022: https://www.societyforscience.org/press-release/regeneron-isef-full-awards-2022/
- Directory finalisti ISEF 2021: https://sspcdn.blob.core.windows.net/files/Documents/SEP/ISEF/2021/Directories/Finalist_Judging.pdf
- EUCYS Leiden 2022, pagina progetto "Year 5": https://eucysleiden2022.eu/year-5-developing-a-novel-multiple-linear-regression-model-to-optimize-honey-bee-gut-immunity-using-a-lactic-acid-bacteria-probiotic-mixture/
- Society for Science blog, EUCYS 2022: https://www.societyforscience.org/blog/experiencing-stem-and-stroopwafels-at-2022-eucys/
- Bee Culture, "Optimizing Bee Gut Immunity": https://beeculture.com/optimizing-bee-gut-immunity/
- Inverse, "Orlando Teen Varun Madan Uses Probiotic Bacteria to Save the Honeybees" (2018): https://www.inverse.com/article/50242-can-bacteria-save-the-bees-young-innovators
- Stanford School of Engineering, "Spotlight: Varun Madan": https://engineering.stanford.edu/spotlight/varun-madan
- Broadcom MASTERS 2018 Finalists: https://www.societyforscience.org/broadcom-masters/broadcom-masters-2018-finalists/
- Broadcom MASTERS 2018 Top 300: https://sspcdn.blob.core.windows.net/files/Documents/SEP/BCM/2018/Program-Books/Top-300-MASTERS.pdf
- Orlando Science Center, Dr. Nelson Ying Science Competition: https://www.osc.org/local-teen-scientists-compete-to-save-the-world-at-dr-nelson-ying-science-competition/
- GitHub, profilo dell'autore (nessun repository sul progetto): https://github.com/madanva
- GitHub, ricerca repository "Nosema honey bee" (3 repo di terzi, non collegati): https://github.com/search?q=Nosema+honey+bee&type=repositories
