# Synthesis of a Soy Protein Hydrogel for Invasive Snail Control in Agricultural Settings (2017)

> Scheda di studio — ISEF 2017 (Los Angeles, 14-19 maggio 2017). Progetto pid 2017-0027, ProjectId 9099 nel database abstract della Society. Tier: **Best of Category** (Animal Sciences).
>
> Nota metodologica: durante questa ricerca il budget di WebSearch della sessione si è esaurito dopo 4 interrogazioni effettive (altre 6 sono state rifiutate dal sistema) e l'egress proxy ha bloccato ssefflorida.com, societyforscience.org, science-fair.org, PMC, IFAS e il Program Book 2017. Le informazioni provengono quindi da: (1) l'abstract ufficiale, già raccolto nel dataset locale a partire da abstracts.societyforscience.org; (2) i riassunti restituiti dalle 4 ricerche web; (3) il file locale `isef_research/data/awards_web/2017_official.json`, costruito in un passo precedente del workflow sulle pagine ufficiali; (4) 6 interrogazioni all'API GitHub + 2 profili GitHub letti direttamente. Tutto ciò che non è stato trovato è marcato "non trovato".

## Scheda

| Campo | Valore |
|---|---|
| Anno | 2017 (Intel ISEF, Los Angeles Convention Center, 14-19 maggio 2017; circa 1.778 finalisti da 78 paesi) [web: https://www.societyforscience.org/press-release/intel-international-science-and-engineering-fair-2017-grand-award-winners/] |
| Premi (tutti, con importi) | **Intel ISEF Best of Category Award, Animal Sciences — $5,000** [abstract/dataset ufficiale]; **First Award — $3,000** nella categoria Animal Sciences (ogni vincitore Best of Category 2017 riceveva anche il First Award) [web: https://www.societyforscience.org/press-release/intel-international-science-and-engineering-fair-2017-grand-award-winners/ ; https://ssefflorida.com/congratulations-florida-intel-international-science-engineering-fair-2017-grand-award-winners/]; in più, secondo il comunicato ufficiale, per ogni Best of Category un **grant Intel Foundation di $1,000 alla scuola** e **$1,000 alla fiera regionale affiliata** [web: comunicato Society 2017, via file locale 2017_official.json]. Totale allo studente: $8,000. |
| Categoria | Animal Sciences (ANIM) |
| Finalista/i | Jessica Young (progetto individuale) |
| Scuola | Palm Beach Central High School [web: ssefflorida.com, societyforscience.org] |
| Luogo | Wellington, Florida, USA [web: 2017_official.json, fonti societyforscience.org e ssefflorida.com]; fiera affiliata: State Science and Engineering Fair of Florida (SSEF) tramite la fiera di contea di Palm Beach (dedotto dalla pagina "Congratulations Florida" di SSEF Florida; nome esatto della fiera regionale: non trovato) |
| Mentore / laboratorio | non trovato (nessun comunicato, intervista o paper reperito; il progetto è compatibile con un laboratorio scolastico di ricerca con vasche per gasteropodi) |
| Codice pubblico | nessun codice pubblico trovato (6 ricerche API GitHub: "soy protein hydrogel snail papain", "apple snail Pomacea", "papain snail", "molluscicide hydrogel", "\"apple snail\"", "soy protein hydrogel" → 0 repository pertinenti; 2 profili "Jessica Young" letti: jessicayoung3 e JessAYoung, nessuna attinenza) |
| Pubblicazioni | non trovate (nessun paper, poster o brevetto reperito nelle ricerche effettuate) |
| Tipo di ricerca | Sperimentale di laboratorio, bio-saggio di tossicità con disegno dose-risposta + test su specie non bersaglio; secondo anno di un percorso sui gasteropodi *Pomacea* (progetto ISEF 2016 della stessa studentessa, vedi sotto) |

## Abstract ufficiale (EN)

[abstract] Testo ufficiale, ProjectId 9099 (abstracts.societyforscience.org):

> The island apple snail (Pomacea maculata) is one of the world's most invasive species that poses environmental and agricultural threats. In rice and taro fields, the snails can cause decreases in crop yields and quality, and expose plants to secondary infections. The enzyme papaya proteinase I (papain) can be used as a biochemical control for nonnative snails in laboratory experiments, however this is not replicatable in the field because the enzyme degrades. To resolve this, 20% (w/v) soy protein hydrogels were created by mixing soy protein isolate with distilled water, adding no papain, 0.5g, 1.0g, 2.5g and 5.0g of papain, and then heating all hydrogels at 60°C for 90 minutes. The efficacy of these hydrogels in inducing mortality in invasive snails was determined by exposing groups of snails to the hydrogels for 120 hours. It was observed that groups exposed to hydrogels containing 2.5g or greater experienced 100% mortality, while groups exposed to 1.0g experienced 75% mortality, and groups exposed to 0.5g experienced 55% mortality. A one-way ANOVA was run to determine that these were statistically significant increases when compared to the control. To further test the hydrogels, groups of 20 ghost shrimp (Palaeomonetes paludosus) were exposed to the various concentrations of papain used. It was found that after 72 hours, no shrimp in any of the groups had died after exposure, indicating that other non-mollusc macroinvertebrates were not affected by the presence of the enzyme. The biodegradable soy protein hydrogels developed in this study can provide economically feasible and environmentally sound ways to eradicate invasive snail populations in agricultural areas and provide a framework for further refinement and eventual usage in environmental settings as well.

Progetto precedente della stessa finalista (ISEF 2016, ProjectId 11559, nessun premio; nel dataset locale la categoria risulta "Systems Software", ma è un artefatto di scraping che colpisce 30 progetti 2016 chiaramente biologici — la categoria reale è con ogni probabilità Animal Sciences): **"The Effects of Potassium Chloride and Experimental Temperature on Florida Apple Snail Development and Exhibition of Predatory Avoidance Behavior"**. [abstract 2016] Studiava la chiocciola nativa *Pomacea paludosa* delle Everglades (preda del nibbio delle Everglades *Rostrhamus sociabilis*): allevamento a KCl ≥ 15 ppt e a temperature sotto i 26,7 °C → crescita rallentata o perdita di massa e ridotta risposta ai kairomoni di predazione. Questo spiega la competenza con l'allevamento di *Pomacea* in acquario e l'attenzione alla specie nativa.

## Problema e domanda di ricerca

[abstract] *Pomacea maculata* (island apple snail, famiglia Ampullariidae) è tra le specie più invasive al mondo; nelle risaie e nei campi di taro riduce resa e qualità e apre la pianta a infezioni secondarie. È noto, da esperimenti di laboratorio, che l'enzima **papaina** (papaya proteinase I, una cisteina-proteasi estratta dal lattice di *Carica papaya*) agisce come controllo biochimico sulle chiocciole non native; però in campo l'enzima **si degrada** (idrolisi, diluizione, degradazione microbica, perdita di attività), quindi il risultato di laboratorio non è replicabile.

Domanda di ricerca (ricostruita dall'abstract): *un idrogel biodegradabile a base di proteina di soia può fungere da matrice/vettore che protegge la papaina e la rilascia in modo da uccidere P. maculata, senza danneggiare altri macroinvertebrati non molluschi?*

Ipotesi implicite: (1) la mortalità delle chiocciole cresce con la dose di papaina nell'idrogel (dose-risposta); (2) l'effetto è selettivo rispetto a crostacei d'acqua dolce (gambero fantasma *Palaemonetes paludosus*, nell'abstract scritto "Palaeomonetes").

## Metodo passo per passo

Tutti i passi seguenti sono [abstract] salvo indicazione.

1. **Preparazione della matrice**: proteina di soia isolata (SPI) dispersa in acqua distillata al **20 % (w/v)** (20 g per 100 mL).
2. **Caricamento dell'enzima**: cinque formulazioni con **0 g (controllo), 0,5 g, 1,0 g, 2,5 g, 5,0 g** di papaina aggiunti alla dispersione. Il volume di idrogel per ogni dose non è riportato (non trovato), quindi la concentrazione effettiva in g/L non è ricostruibile.
3. **Gelificazione termica**: riscaldamento di tutti gli idrogel a **60 °C per 90 minuti**. Commento tecnico (conoscenza di dominio, non dall'abstract): la gelificazione classica dell'SPI avviene per denaturazione delle globuline β-conglicinina (7S, ~70 °C) e glicinina (11S, ~90 °C); 60 °C è sotto entrambe le soglie, quindi si ottiene un gel debole/pasta ad alta concentrazione proteica piuttosto che un gel covalente forte. La scelta di 60 °C è però coerente con la stabilità della papaina, attiva fino a ~65-70 °C: la temperatura serve a legare la matrice senza inattivare l'enzima.
4. **Bio-saggio sulle chiocciole**: gruppi di *P. maculata* esposti per **120 ore** (5 giorni) a ciascun idrogel; registrazione della mortalità per gruppo. Numero di chiocciole per gruppo e repliche: non trovato (vedi inferenza in "Dati").
5. **Statistica**: ANOVA a una via sulle mortalità dei gruppi trattati vs controllo; esito "statisticamente significativo" (valori F, p, post-hoc: non trovati).
6. **Test su specie non bersaglio**: gruppi di **20 gamberi fantasma** (*Palaemonetes paludosus*) esposti alle stesse concentrazioni di papaina per **72 ore**; conteggio dei morti.
7. **Conclusione applicativa**: l'idrogel di soia è biodegradabile, economico e proposto come "framework" da raffinare per uso in campo.

## Tecniche, strumenti e software e codice

- **Materiali** [abstract]: soy protein isolate, acqua distillata, papaina (grado e attività enzimatica in unità non riportati: non trovato), vasche/contenitori per chiocciole e gamberi.
- **Strumentazione** (dedotta): bagno termostatico o incubatore a 60 °C, bilancia, vetreria graduata; strumenti di controllo dell'acqua (temperatura, pH) non citati — non trovato.
- **Tecniche**: gelificazione termica di una proteina vegetale come sistema di rilascio controllato di un enzima; bio-saggio di mortalità a tempo fisso (120 h); saggio di tossicità acuta su non-target (72 h).
- **Software**: non trovato. L'ANOVA a una via è compatibile con Excel (Analysis ToolPak), Minitab o JMP, strumenti tipici dei programmi di ricerca scolastici della Florida; nessuna indicazione nell'abstract.
- **Codice**: nessun codice pubblico trovato. Non esiste componente software nel progetto: l'intero "calcolo" è un'ANOVA su percentuali di mortalità. Se la studentessa avesse scritto codice, avrebbe contenuto: tabella gruppi × dose × vivi/morti, calcolo delle percentuali, ANOVA (F, p) ed eventuale post-hoc Tukey; nulla di più.

## Dati, campioni e statistica

- **Fattore**: dose di papaina a 5 livelli (0 / 0,5 / 1,0 / 2,5 / 5,0 g) [abstract].
- **Variabile di risposta**: mortalità % del gruppo a 120 h (chiocciole) e a 72 h (gamberi) [abstract].
- **Dimensione campionaria**: per i gamberi "groups of 20" [abstract]. Per le chiocciole non trovato; le percentuali 55 % e 75 % sono compatibili con 11/20 e 15/20, quindi è *plausibile* (inferenza, non confermata) che anche i gruppi di chiocciole fossero da 20 individui. Numero di repliche per dose: non trovato.
- **Durata**: 120 h per le chiocciole, 72 h per i gamberi [abstract] — una discrepanza che un giudice noterebbe.
- **Test**: ANOVA a una via, trattati vs controllo, risultato significativo [abstract]; statistiche di test non riportate (non trovato).
- **Osservazione critica**: la mortalità individuale è una variabile binaria; un'ANOVA su percentuali richiede più repliche per dose e ignora la natura binomiale dei dati. Test più appropriati: chi-quadrato o Fisher esatto per tabelle dose × esito, regressione logistica/probit per stimare LC50 e pendenza dose-risposta, Kaplan-Meier se la mortalità fosse stata registrata nel tempo.

## Risultati chiave (numeri)

Tutti [abstract]:

| Dose papaina nell'idrogel | Mortalità *P. maculata* a 120 h | Mortalità *P. paludosus* (gamberi, n = 20) a 72 h |
|---|---|---|
| 0 g (controllo) | riferimento (valore non riportato; implicitamente bassa) | 0 % |
| 0,5 g | **55 %** | 0 % |
| 1,0 g | **75 %** | 0 % |
| 2,5 g | **100 %** | 0 % |
| 5,0 g | **100 %** | 0 % |

- Relazione dose-risposta monotona con saturazione a ≥ 2,5 g.
- ANOVA a una via: incrementi significativi rispetto al controllo (p non riportato).
- Nessuna mortalità nei crostacei a nessuna dose → evidenza (limitata) di selettività verso i molluschi.

## Perché ha vinto — analisi secondo i criteri ISEF

Criteri Science (ISEF): Research Question 10 %, Design & Methodology 15 %, Execution 20 %, Creativity 20 %, Presentation 35 % (poster 10 % + colloquio 25 %).

- **Research Question (10 %)** — Problema reale e quantificabile per l'agricoltura (le *Pomacea* sono tra i peggiori parassiti del riso in Asia e invasori riconosciuti in Florida, Texas, Louisiana), con un gap tecnico preciso: "funziona in laboratorio ma non in campo perché l'enzima si degrada". La domanda nasce da una limitazione nota e la aggredisce direttamente. Punteggio presumibilmente alto.
- **Design & Methodology (15 %)** — Disegno a 5 livelli di dose con controllo a dose zero, durata fissa, variabile di risposta oggettiva (morte). Aggiunge un secondo esperimento su specie non bersaglio: è la mossa che trasforma un semplice "funziona" in "funziona ed è ambientalmente accettabile", cioè la domanda che qualunque giudice di Animal Sciences avrebbe fatto.
- **Execution (20 %)** — Dati puliti e facilmente comunicabili (55/75/100/100 %), trend monotono, test statistico dichiarato, zero mortalità nel non-target. Materiali economici e procedura riproducibile in qualsiasi laboratorio scolastico.
- **Creativity (20 %)** — L'idea di usare un **idrogel proteico biodegradabile come vettore a rilascio prolungato per un enzima molluscicida** è una vera ricombinazione: le proteine vegetali gelificate sono note nella scienza alimentare, la papaina come molluscicida è nota in laboratorio, ma il loro accoppiamento per risolvere il problema di degradazione in campo è il contributo originale. Si nota che in letteratura un'idea analoga (molluscicida a rilascio prolungato in gel di gelatina contro *Pomacea canaliculata*) è stata pubblicata solo anni dopo (titolo visto nei risultati di ricerca: "A Novel Gelatin-Based Sustained-Release Molluscicide for Control of the Invasive Agricultural Pest and Disease Vector Pomacea canaliculata", PMC9268488, non letto perché il dominio è bloccato).
- **Presentation (35 %)** — Non verificabile direttamente, ma la studentessa era al secondo ISEF consecutivo con lo stesso genere animale (2016: *P. paludosa*; 2017: *P. maculata*): due anni di allevamento e osservazione di chiocciole rendono il colloquio molto solido (conoscenza della biologia della specie, delle Everglades, del nibbio delle Everglades, della distinzione nativa/invasiva). L'abstract è chiaro, con numeri espliciti e una conclusione applicativa misurata ("framework for further refinement").

In sintesi: ha vinto perché ha combinato **problema agricolo concreto + gap tecnico chiaro + soluzione ingegneristica semplice + dose-risposta + controllo del rischio ambientale**, raccontati con numeri e con la continuità di un percorso pluriennale.

## Punti deboli e domande da giudice

1. **Meccanismo d'azione non spiegato**: la papaina uccide per ingestione (le chiocciole mangiano il gel proteico come esca?), per contatto con l'epitelio/muco, o per degradazione del muco protettivo? L'abstract non lo dice.
2. **Dose espressa in grammi, non in attività enzimatica (unità/mg)**: la papaina commerciale varia molto; senza unità il risultato non è trasferibile. E il volume dell'idrogel non è riportato, quindi non si conosce la concentrazione.
3. **La papaina digerisce la stessa matrice di soia**: una proteasi inserita in un gel proteico lo idrolizza; quanto dura il gel in acqua? Qual è la cinetica di rilascio? Nessuna misura di rilascio o di attività residua nel tempo (il problema di partenza era proprio la degradazione).
4. **Nessuna prova in condizioni di campo**: niente acqua di risaia, sedimento, temperatura variabile, microbi; la "replicabilità in campo" resta ipotesi.
5. **Statistica**: ANOVA su percentuali binarie; repliche e n non riportati; mancano p, F, intervalli di confidenza, LC50.
6. **Non-target limitato**: una sola specie (crostaceo), 72 h invece di 120 h; non testati pesci, anfibi, piante di riso, né soprattutto la **chiocciola nativa *P. paludosa*** (preda del nibbio delle Everglades, specie protetta): una proteasi non distingue tra molluschi nativi e invasivi. Questa domanda è particolarmente pungente dato il progetto 2016 della stessa studentessa.
7. **Confondenti nel controllo**: 20 % di proteina in decomposizione in acqua ferma può alzare ammoniaca e abbassare l'ossigeno; serviva un controllo "solo acqua" oltre a "gel senza papaina", con misure di qualità dell'acqua.
8. **Scalabilità ed economia**: costo per ettaro, modalità di distribuzione, degradazione in presenza di luce UV e pioggia: non quantificati.

## Percorso dello studente dopo ISEF

Non trovato. Nessuna notizia reperita su università, pubblicazioni, premi successivi o startup (le ricerche nominative restituiscono omonimi non correlati; i profili GitHub omonimi letti non hanno attinenza). Dato certo: due partecipazioni consecutive a ISEF (2016 senza premio, 2017 Best of Category).

## Lezioni per un progetto EAEV/PHYS su radon e bradisismo

1. **Costruire su più anni**: il 2016 (chiocciola nativa, stress ambientale) ha preparato il 2017 (chiocciola invasiva, controllo). Per il radon: un anno di misure di fondo e calibrazione del modello di trasporto (emanazione, diffusione, advezione), un secondo anno di applicazione all'unrest dei Campi Flegrei/Ischia.
2. **Partire da un gap "laboratorio vs campo"**: Young ha attaccato il motivo per cui una tecnica nota fallisce in campo. Analogo radon: le anomalie di Rn-222 in campo sono mascherate da pioggia, umidità del suolo, pressione barometrica, temperatura. Un modello fisico che **rimuove i confondenti meteorologici** (equazione di diffusione-advezione con sorgente ed emanazione dipendenti dall'umidità) è il "vettore idrogel" del vostro progetto: ciò che rende il segnale usabile fuori dal laboratorio.
3. **Disegno a gradiente, non a due punti**: 5 dosi hanno dato una curva, non un confronto sì/no. Per il radon: profili a più profondità, più siti lungo un transetto Solfatara-Pisciarelli o lungo le faglie di Ischia, analisi di sensibilità ai parametri (porosità, coefficiente di diffusione effettivo, velocità di Darcy). Sfruttare la lunghezza di diffusione L = √(D_e/λ): per Rn-222 (λ = 2,1·10⁻⁶ s⁻¹) L ≈ 1-2 m nel suolo, per Rn-220 (λ = 1,25·10⁻² s⁻¹) L ≈ pochi cm; il rapporto Rn-220/Rn-222 distingue sorgente superficiale da trasporto advettivo profondo, cioè la vostra "curva dose-risposta".
4. **Includere un test "non bersaglio"**: il controllo sui gamberi è stato decisivo per la credibilità. Per il radon: un sito di controllo fuori dall'area di unrest, o periodi senza sismicità, per mostrare che il modello **non** produce falsi allarmi; confrontare con sollevamento GPS/InSAR e sismicità INGV-Osservatorio Vesuviano.
5. **Numeri nitidi e statistica adeguata**: "55/75/100 %" si ricordano; ma preparatevi alla domanda sulla statistica. Usate test adatti alla natura dei dati (serie temporali: correlazione incrociata con ritardo, test di cambio di regime, bootstrap) e riportate n, intervalli di confidenza e incertezze strumentali (es. RadonEye RD200 ±10 %, camere a scintillazione tipo Lucas, rivelatori a stato solido CR-39).
6. **Spiegare il meccanismo**: il punto debole maggiore del progetto Young era il "come". Per il radon dovete poter scrivere alla lavagna ∂C/∂t = D_e ∂²C/∂z² − v ∂C/∂z − λC + S e dire cosa succede a ogni termine quando sale la pressione dei fluidi idrotermali.
7. **Ancoraggio a uno stakeholder**: agricoltori per Young; Protezione Civile e INGV per voi. Chiudete con "framework for further refinement", non con "abbiamo previsto l'eruzione".
8. **Materiali economici e riproducibili**: soia + papaina. Per voi: rete di rivelatori a basso costo + modello open-source su GitHub con dati e notebook — avreste, a differenza di Young, un codice pubblico da mostrare, che pesa su Execution e Creativity.

## Fonti

Fonti effettivamente consultate (direttamente o tramite riassunti dei motori di ricerca):

- Abstract ufficiale ISEF 2017, ProjectId 9099 (raccolto nel dataset locale a partire da): https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=0%2C&Category=Any%20Category&AllAbstracts=True&FairCountry=Any%20Country&FairState=Any%20State&ProjectId=9099
- Abstract ufficiale ISEF 2016 della stessa finalista, ProjectId 11559: https://abstracts.societyforscience.org/Home/FullAbstract?ISEFYears=0%2C&Category=Any%20Category&AllAbstracts=True&FairCountry=Any%20Country&FairState=Any%20State&ProjectId=11559
- Society for Science, "Intel International Science and Engineering Fair 2017 Grand Award Winners" (premio, scuola, First Award $3,000; letto tramite riassunto di ricerca e file locale 2017_official.json; fetch diretto bloccato): https://www.societyforscience.org/press-release/intel-international-science-and-engineering-fair-2017-grand-award-winners/
- SSEF Florida, "Congratulations Florida – Intel ISEF 2017 Grand Award Winners" (scuola Palm Beach Central High School; fetch diretto bloccato): https://ssefflorida.com/congratulations-florida-intel-international-science-engineering-fair-2017-grand-award-winners/
- science-fair.org, "2017 Intel ISEF Winners" (solo elenco nei risultati di ricerca; fetch bloccato): https://science-fair.org/2017-intel-isef-winners/
- Society for Science, pagina evento Intel ISEF 2017: https://www.societyforscience.org/isef/intel-isef-2017/
- File locale di sintesi dei premi ufficiali 2017: `isef_research/data/awards_web/2017_official.json`
- API GitHub (ricerche repository, tutte senza risultati pertinenti): https://api.github.com/search/repositories?q=soy+protein+hydrogel+snail+papain ; https://api.github.com/search/repositories?q=apple+snail+Pomacea ; https://api.github.com/search/repositories?q=papain+snail ; https://api.github.com/search/repositories?q=molluscicide+hydrogel ; https://api.github.com/search/repositories?q=%22apple+snail%22 ; https://api.github.com/search/repositories?q=soy+protein+hydrogel
- API GitHub (ricerca utenti): https://api.github.com/search/users?q=Jessica+Young ; profili letti senza attinenza: https://github.com/jessicayoung3 ; https://github.com/JessAYoung
- Letteratura correlata vista solo per titolo nei risultati di ricerca (non letta, domini bloccati): "A Novel Gelatin-Based Sustained-Release Molluscicide for Control of the Invasive Agricultural Pest and Disease Vector Pomacea canaliculata", https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9268488/ ; UF/IFAS EENY323/IN598 "Applesnails of Florida Pomacea spp.", https://ask.ifas.ufl.edu/publication/IN598
