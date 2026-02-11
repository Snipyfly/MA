# Expose-Kurzfassung (Anmeldung Masterarbeit, 2-3 Seiten)

## 1. Arbeitstitel (vorlaeufig)
**Datengetriebene Flankenmuster im Profifussball:**  
Haeufigkeit, Effektivitaet und bedingte Wirkung in Bundesliga und 2. Bundesliga

## 2. Ausgangslage und Relevanz
Flanken sind ein zentraler Bestandteil des Offensivspiels, werden in der Forschung jedoch oft nur aggregiert betrachtet oder in wenige vordefinierte Kategorien eingeteilt. Dadurch bleibt offen, welche Flankenmuster im realen Ligabetrieb tatsaechlich auftreten und welche Muster unter welchen Bedingungen wirksam sind.  
Insbesondere fehlen Studien, die
1. Flankenmuster zunaechst datengetrieben identifizieren,
2. diese Muster auf Abschlusswahrscheinlichkeit und Abschlussqualitaet pruefen und
3. Spielzustand, Strafraumdichte und Formation gemeinsam als Kontextfaktoren beruecksichtigen.

Die Arbeit adressiert diese Luecke mit einem grossen Event-Datensatz aus Bundesliga und 2. Bundesliga ueber drei Saisons.

## 3. Zielsetzung, Forschungsfragen und Hypothesen
### 3.1 Zielsetzung
Ziel ist die empirische Rekonstruktion eines robusten Flankenmuster-Raums sowie die Analyse, welche Muster haeufig genutzt werden und unter welchen Bedingungen sie effektiv sind.

### 3.2 Forschungsfragen
**Hauptforschungsfrage:**  
Welche datengetrieben ermittelten Open-Play-Flankenmuster treten in Bundesliga und 2. Bundesliga (Saisons 2022/2023 bis 2024/2025) am haeufigsten auf und welche Muster sind am effektivsten hinsichtlich Abschlusswahrscheinlichkeit und Abschlussqualitaet?

**Sekundaere Moderatorfrage:**  
Variieren diese Muster-Effekte systematisch zwischen Wingback-Systemen (`3er/5er`) und Viererkettensystemen (`4er`)?

### 3.3 Hypothesen
**H1 (Nutzung):**  
Die Nutzung der datengetrieben identifizierten Flankenmuster unterscheidet sich zwischen `winning`, `drawing` und `losing`.

**H2 (Bedingte Effektivitaet):**  
Die Effektivitaet der Flankenmuster variiert in Abhaengigkeit von `MatchStateForTeam` und `box_crowded`.

**H3 (Formation als Moderator):**  
Die Effekte der Flankenmuster auf `shot_within_8s` und `ShotxG` werden durch `formation_macro` (`wingback` vs `back4`) moderiert.

## 4. Datenbasis und Operationalisierung
### 4.1 Datenbasis
Die Analyse basiert auf bereits aufbereiteten Eventdaten:
1. `N=34.604` Open-Play-Flanken,
2. `1.836` Matches,
3. Bundesliga und 2. Bundesliga,
4. Saisons 2022/2023, 2023/2024, 2024/2025,
5. Formationsabdeckung ohne fehlende `LineUp`-Werte (QA: `0.0%` Missing).

### 4.2 Untersuchungseinheit und Endpunkte
Untersuchungseinheit ist das Flanken-Event (`EventId`) mit Raum-, Kontext- und Outcome-Information.

**Primaerer Endpunkt:** `shot_within_8s` (binar, Schuss innerhalb 8 Sekunden).  
**Sekundaerer Endpunkt:** `ShotxG` (nur fuer Events mit Schuss).

### 4.3 Kernvariablen
**Mustermerkmale (Clusterinput):**  
`abs_x`, `abs_y`, `abs_x_rec`, `abs_y_rec`, `delta_abs_x`, `delta_abs_y`, `switch_side`, `MaxHeight`, `NumAttInBox`, `NumDefInBox`, `box_balance`.

**Bedingungen/Kontext:**  
`MatchStateForTeam`, `box_crowded`, `box_balance`.

**Moderator:**  
`formation_macro` aus `LineUp` als `wingback` (`3er/5er`) vs `back4` (`4er`).

## 5. Methodisches Vorgehen
### 5.1 Musteridentifikation
Die Flankenmuster werden datengetrieben mittels `k`-Means bestimmt. Vor der Clusterung erfolgen medianbasierte Imputation und z-Standardisierung numerischer Variablen.

### 5.2 K-Entscheidungslogik (methodisch sauber)
Die Wahl von `K` erfolgt ueber drei etablierte Kriterien:
1. Elbow-Methode (Inertia),
2. Silhouette-Score,
3. Gap Statistic (inkl. 1-SE-Regel).

Gepruefter Bereich: `K=3..10`.  
Fuer die finale Auswahl gelten zusaetzliche Nebenbedingungen:
1. `K>=5` (inhaltlich differenzierte Musterdarstellung),
2. Mindest-Clusteranteil `>=3%` (keine Kleinstcluster als Hauptkategorien).

Resultat im zulaessigen Bereich `K={5,6,7,8,9}`:
1. Elbow: `K=7`
2. Silhouette: `K=5`
3. Gap: `K=9`

Da keine Mehrheitsentscheidung vorliegt, wird als Tie-Break die hoechste Silhouette unter den tie-Kandidaten verwendet. Finale Entscheidung: **`K=5`**.

### 5.3 Cluster unterscheiden und benennen
Die Cluster werden entlang fuenf Achsen interpretiert:
1. Starttiefe (`frueh` / `zwischenraum` / `grundlinie`),
2. Breite (`zentral` / `halbraum` / `breit`),
3. Trajektorie (`vorwaerts` / `neutral` / `rueckraum`),
4. Hoehe (`flach` / `mittel` / `hoch`),
5. Seitenmodus (`gleichseite` / `seitenwechsel`).

Darauf basieren regelgeleitete Arbeitsnamen:
1. `P1`: Grundlinie-Halbraum-Neutral-Typ (`27.89%`, `ShotRate=0.198`)
2. `P2`: Fruehe Fluegelhereingabe (`24.14%`, `ShotRate=0.202`)
3. `P3`: Rueckraumorientierte Flanke (`18.77%`, `ShotRate=0.049`)
4. `P4`: Hohe Seitenwechsel-Flanke (`14.83%`, `ShotRate=0.245`)
5. `P5`: Hoher Grundlinien-Rueckraumball (`14.37%`, `ShotRate=0.352`)

Diese Benennung ist bewusst als **Arbeitsklassifikation** angelegt und kann im Diskussionsteil theoretisch nachgeschaerft werden.

### 5.4 Inferenzanalyse
Auf Basis der Clusterzuordnung folgen zwei Modelle:
1. **Modell A (logistisch):** `P(shot_within_8s=1)` mit Interaktionen `Muster x MatchStateForTeam`, `Muster x box_crowded`, `Muster x formation_macro`.
2. **Modell B (linear):** `ShotxG | shot_within_8s=1` mit analoger Interaktionsstruktur.

Damit werden sowohl Nutzungsunterschiede als auch bedingte Effektivitaetseffekte pruefbar.

## 6. Erwarteter wissenschaftlicher Beitrag
Die Arbeit liefert einen empirisch hergeleiteten Flankenmuster-Raum fuer den deutschen Profifussball und verbindet explorative Musterentdeckung mit hypothesengeleiteter Wirkungsanalyse.  
Der Mehrwert liegt in:
1. datengetriebener statt rein vordefinierter Typologie,
2. gleichzeitiger Betrachtung von Wahrscheinlichkeit (`shot_within_8s`) und Qualitaet (`ShotxG`),
3. expliziter Kontextualisierung ueber Spielstand, Strafraumdichte und Formation.

## 7. Limitationen
1. Formationen liegen als statische Match-Information vor; Ingame-Wechsel sind nicht dynamisch modelliert.
2. Cluster sind feature- und datensatzabhaengig; externe Replikation ist erforderlich.
3. Outcome-Fenster (`6s/8s/10s`) haengt von der Event-Linking-Logik ab.

## 8. Vorlaeufiger Arbeits- und Kapitelplan
### 8.1 Kapitelstruktur (vorlaeufig)
1. Einleitung und Problemstellung
2. Forschungsstand und theoretischer Rahmen
3. Daten, Variablen und Operationalisierung
4. Methodik (Clusterung, K-Entscheidung, Modelle)
5. Ergebnisse
6. Diskussion, Limitationen und Implikationen
7. Fazit

### 8.2 Arbeitsplan (kompakt)
1. **Phase 1:** Finalisierung Expose und Literaturmatrix  
2. **Phase 2:** Hauptanalysen und Robustheitschecks  
3. **Phase 3:** Ergebnisdarstellung und Diskussion  
4. **Phase 4:** Finales Writing und formale Abgabevorbereitung

## 9. Verwendete Kernoutputs (Projektordner)
1. `03h_pattern_k_entscheidung.csv` (K-Entscheidung, transparent dokumentiert)
2. `03b_pattern_selection_metrics.csv` (Kennzahlen je K)
3. `03e_pattern_cluster_profiles.csv` (Clusterprofile)
4. `03f_pattern_namensvorschlaege.csv` (Arbeitsnamen)
5. `03g_pattern_unterschiede_summary.csv` (trennende Merkmale je Cluster)
6. `07_expose_textbausteine.md` (ausfuehrliche Textfassung)
