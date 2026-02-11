# Exposé-Entwurf (BA-Stil) für die Masterarbeit

## Arbeitstitel (vorläufig)
**Effektiv Tore schießen?**  
Eine datengetriebene Analyse von Flankenmustern und ihrer Effektivität im professionellen Fußball

## 1. Ausgangslage und Relevanz des Themas
Die datenanalytische Untersuchung von Offensivmustern hat im professionellen Fußball in den letzten Jahren deutlich an Bedeutung gewonnen. Insbesondere die Frage, welche Angriffsformen unter realen Wettkampfbedingungen tatsächlich zu Torchancen und Toren führen, ist für Vereine, Trainerstäbe und Spielanalysten zentral. Ein wiederkehrender Befund der Literatur ist dabei, dass Flanken trotz ihrer strategischen Bedeutung häufig pauschal als eine einheitliche Aktionsklasse betrachtet werden. Gleichzeitig zeigen empirische Arbeiten, dass die Erfolgswahrscheinlichkeit von Flanken stark variieren kann und vom situativen Kontext abhängt (z. B. Pulling et al., 2018; Mitrotasios et al., 2019; Vecer, 2013).

Aus Perspektive der Praxis ist diese Vereinfachung problematisch: Eine Flanke von der Grundlinie in den Rückraum unterscheidet sich taktisch und in ihrer erwartbaren Wirkung deutlich von einer frühen Hereingabe aus tieferen Zonen oder einer hohen Seitenwechsel-Flanke. Wenn diese Varianten analytisch nicht getrennt werden, bleiben zentrale Unterschiede in Häufigkeit, Risiko und Effektivität unsichtbar. Für die Trainings- und Matchplan-Praxis bedeutet das, dass Entscheidungsregeln für Flankensituationen oft auf aggregierten Kennzahlen beruhen, die strukturell unterschiedliche Aktionen zusammenfassen.

Für die sportwissenschaftliche Forschung ergibt sich daraus eine doppelte Aufgabe: Erstens müssen Flankenmuster datengetrieben rekonstruiert werden, statt sie ausschließlich vorab theoretisch festzulegen. Zweitens ist zu prüfen, wie sich diese Muster unter verschiedenen Bedingungen verhalten, insbesondere in Abhängigkeit vom Spielstand, von der Strafraumbesetzung und von der Grundformation der angreifenden Mannschaft. Die vorliegende Arbeit setzt an genau dieser Schnittstelle zwischen explorativer Musterfindung und hypothesengeleiteter Wirkungsanalyse an.

## 2. Theoretischer Hintergrund, Forschungsstand und Forschungslücke

### 2.1 Historische Bedeutung und aktuelle Relevanz von Flanken
Flanken gelten seit Jahrzehnten als zentrale Offensivaktion im Fußball. Für die Premier League wird berichtet, dass `14,5%` aller Tore aus Open-Play-Flankensituationen entstehen (Vecer, 2013). Historische WM-Analysen verweisen zudem auf teilweise noch höhere Anteilswerte, unter anderem auf `33%` in einer älteren Turnieranalyse (Partridge & Franks, 1989a). Für mehrere Weltmeisterschaften wird ein Korridor von etwa `13% bis 28%` an Toren nach Flanken aus dem Spiel berichtet (Smith & Lyons, 2017).  
Diese Befunde markieren Flanken als relevante Abschlussvorbereitung, zeigen aber zugleich, dass der Effekt je nach Wettbewerb und Kontext stark variiert.

### 2.2 Das Effizienzparadox von Flanken
Parallel zur Relevanz von Flanken wird in der Literatur ein Effizienzparadox beschrieben: Die Flankenanzahl sinkt in Topligen, während die Erfolgswahrscheinlichkeit einzelner Flanken häufig niedrig bleibt. Für die Bundesliga wird beispielsweise ein Rückgang von `12.015` Flanken (2009/2010) auf `8.878` (2015/2016) berichtet (Vecer, 2013).  
Gleichzeitig wird für Open-Play-Flanken eine niedrige Torverwertungsquote ausgewiesen (`1,088%`), die unter alternativen Angriffswegen liegt (Vecer, 2013). Andere Arbeiten berichten etwas höhere, aber ebenfalls begrenzte Quoten (`2,7% bis 3,2%`) (Partridge & Franks, 1989a, 1989b; Pulling et al., 2018). Ebenfalls wird hervorgehoben, dass nur ein kleiner Teil der Flanken überhaupt beim Mitspieler ankommt und ein hoher Anteil in Ballverlusten endet (Vecer, 2013).  
In dieselbe Richtung zeigen Befunde, wonach hohe Flankenvolumina teilweise negativ mit Gewinnwahrscheinlichkeit assoziiert sind (Liu et al., 2015) und kontrolliertere Angriffsformen über kurze Pässe in bestimmten Kontexten vorteilhaft sein können (Oberstone, 2009).

### 2.3 Flanken sind nicht homogen: Typisierung und Raumlogik
Ein zentraler theoretischer Punkt der bestehenden Literatur ist, dass Flanken nicht als einheitlicher Aktionstyp verstanden werden sollten. Frühere Arbeiten zeigen bereits deutliche Wirksamkeitsunterschiede zwischen Flankentypen, etwa in Abhängigkeit von Ausführungsort und Flugprofil (Partridge & Franks, 1989a, 1989b).  
Zur räumlichen Strukturierung werden in der Flankenforschung wiederholt Zonenmodelle verwendet. Für den Flankenausgangsort werden unter anderem fünf Zonen berichtet (Pulling et al., 2018). Für Zielräume wird beschrieben, dass bestimmte Zonen vor dem Tor (u. a. Zone 8) besonders häufig mit Abschlüssen verbunden sind (Pulling et al., 2018; Yamada & Hayashi, 2015). Zudem werden taktische Muster beschrieben, bei denen Teams durch Vororientierung auf einer Seite Räume auf der ballfernen Seite für die Hereingabe öffnen (Hawkins & Robinson, 2016).  
Auch für den Ausgangsort deuten die Befunde auf klare Unterschiede hin: Für einzelne Zonen (u. a. 4 und 10) werden günstigere Ergebnisse berichtet als für weiter entfernte oder ungünstigere Winkelzonen (Pulling et al., 2018; Pollard et al., 2004).

### 2.4 Kontextfaktoren: Spielstand, Defensivdruck und Strafraumbesetzung
Mehrere Arbeiten weisen darauf hin, dass der Erfolg von Flanken stark kontextabhängig ist. Insbesondere der Spielstand wird als relevanter Faktor beschrieben: Führende Teams reduzieren Flanken tendenziell, während zurückliegende Teams häufiger flanken (Gelade, 2017; Vecer, 2013; Zhou et al., 2020).  
Darüber hinaus spielt Defensivdruck eine zentrale Rolle. Für Flanken werden hohe Anteile unter mittlerem bzw. hohem Druck berichtet (Pulling et al., 2018). Taktisch wird dies durch das seitliche Lenken des ballführenden Spielers auf Außenbahnen begründet, um zentrale Räume zu schließen (Bangsbo & Peitersen, 2000; Wein, 2004). Damit beeinflusst Defensivdruck sowohl den Zeitpunkt als auch die Qualität der Flankenabgabe (Pulling et al., 2018; Tenga, 2010).  
Aus theoretischer Sicht folgt daraus, dass Flankenwirkung nur sinnvoll in Verbindung mit Raum- und Besetzungsmerkmalen (z. B. Spieleranzahl im Strafraum) interpretiert werden kann.

### 2.5 Stand der Forschung im Ligakontext
Ein wiederkehrender Befund ist, dass Open-Play-Flanken im Vergleich zu Standardsituationen insgesamt seltener untersucht werden (Pulling et al., 2018). Vorhandene Arbeiten zeigen zwar konkrete Handlungsmuster (z. B. Rücken der Abwehr, Zielräume zwischen Fünfer und Elfmeterpunkt), gleichzeitig bleibt die Evidenz zwischen Ligen und Teamprofilen heterogen (Partridge & Franks, 1989a, 1989b; Yamada & Hayashi, 2015; Mitrotasios et al., 2019; Sarmento et al., 2013).  
Zusätzlich werden widersprüchliche Teammuster beschrieben, etwa dass Teams mit hoher Flankenkompetenz nicht zwingend häufiger flanken (Sarkar, 2018), während andere Studien für bestimmte Ligen signifikante Unterschiede nach Teamranking berichten (González-Rodenas et al., 2023). Parallel gibt es Hinweise, dass Anzahl und Effektivität von Flanken mit Teamerfolg zusammenhängen können (Andrzejewski et al., 2022; Casal et al., 2021; Mitrotasios et al., 2019; Zhou et al., 2021).

### 2.6 Forschungslücke und Ableitung für die Masterarbeit
Auf Basis des vorliegenden Forschungsstands lassen sich für die geplante Masterarbeit vier konkrete Lücken ableiten:
1. **Musterbildung:** Viele Arbeiten analysieren Flanken in vorgegebenen Kategorien statt datengetriebener Musteridentifikation.
2. **Wirkungsmaße:** Die gemeinsame Betrachtung von Abschlusswahrscheinlichkeit und Abschlussqualität (`xG`) bleibt begrenzt.
3. **Kontextintegration:** Spielstand, Druck-/Raumkontext und Strafraumbesetzung werden selten simultan modelliert.
4. **Formation und Datenscope:** Der Formationseinfluss und große Ligastichproben über mehrere Saisons sind unterrepräsentiert.

Genau diese Lücken adressiert die vorliegende Arbeit mit einem Event-Datensatz aus Bundesliga und 2. Bundesliga über drei Saisons, einer datengetriebenen Flankenmusterermittlung und anschließender Modellierung bedingter Effektivität.

## 3. Zielsetzung, Forschungsfragen und Hypothesen
Ziel der Arbeit ist es, einen empirisch belastbaren Flankenmuster-Raum für den deutschen Profifußball zu identifizieren und anschließend die Effektivität dieser Muster unter realen Spielsituationen zu prüfen.

**Hauptforschungsfrage:**  
Welche datengetrieben ermittelten Open-Play-Flankenmuster treten in Bundesliga und 2. Bundesliga (Saisons 2022/2023 bis 2024/2025) am häufigsten auf und welche Muster sind am effektivsten hinsichtlich Abschlusswahrscheinlichkeit und Abschlussqualität?

**Sekundäre Moderatorfrage:**  
Unterscheiden sich diese Muster-Effekte systematisch zwischen Wingback-Systemen (`3er/5er`) und Viererkettensystemen (`4er`)?

Daraus werden folgende Hypothesen abgeleitet:

**H1 (Nutzungsmuster):**  
Die Nutzung der datengetrieben identifizierten Flankenmuster unterscheidet sich signifikant zwischen `winning`, `drawing` und `losing`.

**H2 (Bedingte Effektivität):**  
Die Effektivität von Flankenmustern (gemessen über `P(shot_within_8s)` und `ShotxG`) variiert in Abhängigkeit von `MatchStateForTeam` und `box_crowded`.

**H3 (Formationsmoderation):**  
Die Effekte der Flankenmuster auf Abschlusswahrscheinlichkeit und Abschlussqualität werden durch `formation_macro` (`wingback` vs `back4`) moderiert.

## 4. Methodik

### 4.1 Studiendesign
Die Arbeit folgt einem sequenziellen Design aus:
1. explorativer Musteridentifikation (datengetrieben),
2. deskriptiver Kontextanalyse,
3. inferenzstatistischer Hypothesenprüfung.

Damit wird eine Brücke zwischen unüberwachter Strukturentdeckung und theoriegeleiteter Modellierung geschlagen.

### 4.2 Datenbasis und Untersuchungseinheit
Die Datengrundlage umfasst Open-Play-Flanken aus Bundesliga und 2. Bundesliga für die Saisons 2022/2023 bis 2024/2025. Aktueller Stand der aufbereiteten Daten:
1. `N=34.604` Flanken-Events,
2. `1.836` Matches,
3. vollständige Formationsabdeckung in den verwendeten Events (`ShareWithoutLineUpPct = 0.0`).

Untersuchungseinheit ist das Event (`EventId`). Über Event-Links werden Outcome-Informationen ergänzt.

### 4.3 Variablen und Operationalisierung
**Primärer Endpunkt:** `shot_within_8s` (binär; Schuss innerhalb von 8 Sekunden nach Flanke).  
**Sekundärer Endpunkt:** `ShotxG` (kontinuierlich; nur bedingt auf Schuss-Events).

**Mustermerkmale (Clusterinput):**  
`abs_x`, `abs_y`, `abs_x_rec`, `abs_y_rec`, `delta_abs_x`, `delta_abs_y`, `switch_side`, `MaxHeight`, `NumAttInBox`, `NumDefInBox`, `box_balance`.

**Kontext-/Kontrollmerkmale:**  
`MatchStateForTeam`, `box_crowded`, `Season`, `Competition`.

**Moderator:**  
`formation_macro` aus `LineUp` als `wingback` (`3er/5er`) vs `back4` (`4er`).

### 4.4 Datenaufbereitung
Numerische Variablen werden median-imputiert und z-standardisiert, um Skaleneffekte zwischen räumlichen, besetzungsbezogenen und höhenbezogenen Features zu kontrollieren. Event-IDs werden dedupliziert und konsistent auf die Outcome-Tabellen gejoint. Der aktuelle Datensatz reproduziert eine plausible Gesamtquote von `shot_within_8s` von ca. `0.200`.

### 4.5 Identifikation der Flankenmuster und K-Entscheidung
Die Flankenmuster werden per `k`-Means identifiziert. Die Clusterzahl wird methodisch nicht ad hoc gesetzt, sondern über drei Standardkriterien bestimmt:
1. Elbow (Inertia-Verlauf),
2. Silhouette,
3. Gap Statistic (inkl. 1-SE-Logik).

Geprüft wurde `K=3..10`. Für die finale Exposé-relevante Auswahl wurden zusätzlich Nebenbedingungen gesetzt:
1. `K >= 5` (inhaltlich ausreichende Musterdifferenzierung),
2. Mindest-Clusteranteil `>= 3%` (Stabilität und Interpretierbarkeit).

Ergebnisse im zulässigen Bereich `K={5,6,7,8,9}`:
1. Elbow: `K=7`,
2. Silhouette: `K=5`,
3. Gap: `K=9`.

Da keine Mehrheitsentscheidung vorlag, wurde als transparente Tie-Break-Regel die höchste Silhouette unter den Tie-Kandidaten gewählt. Finale Lösung: **`K=5`**.

Die Entscheidung ist vollständig dokumentiert in  
`/Users/hofmann/PycharmProjects/MA_Statistik/cleaned/expose_outputs/03h_pattern_k_entscheidung.csv`.

### 4.6 Clusterunterscheidung und Benennung
Die inhaltliche Interpretation erfolgt entlang fünf Achsen:
1. Starttiefe,
2. Breite,
3. Trajektorie,
4. Flankenhöhe,
5. Seitenmodus.

Auf dieser Basis wurden aktuell fünf Arbeitsmuster benannt:
1. `P1` Grundlinie-Halbraum-Neutral-Typ (`27.89%`, `ShotRate=0.198`),
2. `P2` Frühe Flügelhereingabe (`24.14%`, `ShotRate=0.202`),
3. `P3` Rückraumorientierte Flanke (`18.77%`, `ShotRate=0.049`),
4. `P4` Hohe Seitenwechsel-Flanke (`14.83%`, `ShotRate=0.245`),
5. `P5` Hoher Grundlinien-Rückraumball (`14.37%`, `ShotRate=0.352`).

Die Namen sind als analytische Arbeitsbezeichnungen zu verstehen und werden in der Diskussion sportpraktisch eingeordnet.

### 4.7 Inferenzstatistische Auswertung
Zur Hypothesenprüfung werden zwei Modelle geschätzt:
1. **Modell A (logistisch):** `P(shot_within_8s=1)`.
2. **Modell B (linear/GLM):** `ShotxG | shot_within_8s=1`.

Beide Modelle enthalten Interaktionen für:
1. `pattern_cluster x MatchStateForTeam`,
2. `pattern_cluster x box_crowded`,
3. `pattern_cluster x formation_macro`.

Zusätzlich werden Liga- und Saisonkontrollen berücksichtigt.

### 4.8 Robustheit und Validierung
Zur Absicherung der Befunde sind vorgesehen:
1. Join-Integritätsprüfung auf Event-Ebene,
2. Sensitivität der Outcome-Definition (`6s`, `8s`, `10s`),
3. Prüfung auf seltene Clusterklassen,
4. Vergleich der Effektmuster über Liga- und Saison-Schnitte.

## 5. Erwarteter Beitrag der Arbeit
Die Arbeit leistet einen methodischen und inhaltlichen Beitrag:
1. **Methodisch:** Kombination aus datengetriebener Musterbildung und transparenter K-Entscheidungslogik.
2. **Inhaltlich:** Evidenz, welche Flankenmuster in welchen Kontexten effektiv sind.
3. **Praxisbezogen:** Ableitung differenzierter Handlungsempfehlungen für Musterwahl statt pauschaler Flankenrate.

## 6. Limitationen
1. Formationen liegen als statische Match-Informationen vor; dynamische Ingame-Wechsel sind nicht direkt modelliert.
2. Clusterlösungen sind daten- und featureabhängig; externe Replikationen bleiben erforderlich.
3. Die Outcome-Logik basiert auf Event-Linking und damit auf den zugrundeliegenden Tracking/Event-Konventionen.

## 7. Vorläufiger Gliederungsplan
1. Einleitung
2. Theoretischer Hintergrund und Forschungsstand
3. Forschungslücke, Forschungsfragen und Hypothesen
4. Daten und Methodik
5. Ergebnisse
6. Diskussion
7. Fazit und Ausblick
8. Literaturverzeichnis
9. Anhang

## 8. Vorläufiger Zeitplan (kompakt)
1. **Phase 1:** Exposé-Finalisierung und Literaturmatrix
2. **Phase 2:** Hauptanalysen und Robustheitschecks
3. **Phase 3:** Ergebnisdarstellung und Diskussion
4. **Phase 4:** Final Writing, Formatierung, Abgabe

## 9. Literatur (Auswahl, im Text verwendet)
Andrzejewski et al. (2022); Bangsbo & Peitersen (2000); Casal et al. (2021); Gelade (2017); González-Rodenas et al. (2023); Hawkins & Robinson (2016); Liu et al. (2015); Mitrotasios et al. (2019); Oberstone (2009); Partridge & Franks (1989a, 1989b); Pollard et al. (2004); Pulling et al. (2018); Sarkar (2018); Sarmento et al. (2013); Smith & Lyons (2017); Tenga (2010); Vecer (2013); Wein (2004); Yamada & Hayashi (2015); Zhou et al. (2020, 2021).
