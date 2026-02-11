# Expose text blocks (auto-generated)

## Forschungsfrage
Welche datengetrieben ermittelten Open-Play-Flankenmuster treten in Bundesliga und 2. Bundesliga in den Saisons 2022/2023, 2023/2024, 2024/2025 am haeufigsten auf, welche Muster sind effektiver in Bezug auf `shot_within_8s` und `ShotxG`, und unter welchen Bedingungen (Spielzustand, Box-Crowding) variieren diese Effekte?

## Sekundaere Moderatorfrage
Unterscheiden sich diese Muster-Effekte systematisch zwischen Wingback-Systemen (`3er/5er`) und Viererketten-Systemen (`4er`)?

## Hypothesen
H1: Die Nutzung der datengetriebenen Flankenmuster unterscheidet sich zwischen Spielzustaenden (`winning`, `drawing`, `losing`).
H2: Die Effektivitaet der Muster ist bedingungsabhaengig von `MatchStateForTeam` und `box_crowded`.
H3: Die Muster-Effekte auf `shot_within_8s` und `ShotxG` werden durch `formation_macro` (`wingback` vs `back4`) moderiert.

## Datengrundlage
- Event-Level Flanken: 34604
- Matches: 1836
- Wettbewerbe: 2. Bundesliga, Bundesliga
- Anteil Flanken mit Schuss <=8s: 0.2002
- Entdeckte Muster (k-means, K=5): P1=9652, P2=8354, P3=6494, P4=5133, P5=4971
- Namensvorschlaege: P1:Grundlinie-Halbraum-Neutral-Typ, P2:Fruehe Fluegelhereingabe, P3:Rueckraumorientierte Flanke, P4:Hohe Seitenwechsel-Flanke, P5:Hoher Grundlinien-Rueckraumball

## Methodik
1. Explorative Musterermittlung mit k-means auf Flankenmerkmalen (`X`, `Y`, `X_rec`, `Y_rec`, `MaxHeight`, `NumAttInBox`, `NumDefInBox`, abgeleitete Raum-/Trajektorienmerkmale).
2. K-Auswahl methodisch ueber drei Standardkriterien: Elbow (Inertia), Silhouette und Gap Statistic; finale K-Entscheidung per Mehrheitsregel unter Nebenbedingungen (`K>=5`, Mindest-Clusteranteil).
3. Deskriptive Auswertung der Musterhaeufigkeit und Muster-Effektivitaet nach Liga, Saison, Spielzustand, Box-Crowding.
4. Modell A: logistische Regression fuer `P(shot_within_8s=1)` mit Interaktionen Muster x Bedingungen und Muster x Formation-Makrogruppe.
5. Modell B: lineares Modell fuer `ShotxG | shot_within_8s=1` mit derselben Interaktionslogik.

## Operationalisierung
- UV-Kern: `pattern_cluster` (datengetrieben, keine vorab fixe Musterliste)
- Bedingungen: `MatchStateForTeam`, `box_crowded`, `box_balance`
- Moderator: `formation_macro` (`wingback` vs `back4`)
- AV1: `shot_within_8s`
- AV2: `ShotxG` (konditional auf Schuss)

## Limitationen
- Cluster sind daten- und feature-abhaengig; externe Replikation in anderen Ligen/Saisons erforderlich.
- Formationen liegen statisch pro Match-Team vor (keine in-game Wechselzeitachse).
- Zeitfenster-Sensitivitaet (6/8/10s) basiert auf Event-Linking-Logik.
