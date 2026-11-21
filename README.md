# FRC Vision Lab 2.0

אפליקציית דסקטופ (Python + OpenCV + Qt) לזיהוי כל ה‑AprilTags של מגרש FRC, מגרש תלת‑ממדי, כיול מצלמה בסגנון Limelight – והעלאת הכיול ישירות לליימלייט.

## הרצה
```
run.bat
```
או: `pip install -r requirements.txt` ואז `python main.py`

## מסכים (Ctrl+1..4)

| מסך | מה יש בו |
|-----|-----------|
| **AprilTags** | זיהוי tag36h11, ‏tx/ty/ta, מרחק ו‑3D לכל tag, ‏botpose ‏(wpiblue) מכמה תגים יחד, **מגרש 3D** (או מפה 2D), בדיקת דיוק מול מטר, הקלטת CSV |
| **Calibration** | כיול ChArUco כמו ב‑Limelight: לוח להדפסה (כולל הלוח הרשמי של Limelight), צילום אוטומטי, כיסוי, **תצוגת 3D של מיקומי הלוח**, RMS / fx / fy / cx / cy / FOV / pixel aspect |
| **Camera** | מצלמת USB או stream (‏Limelight / PhotonVision), רזולוציה, חשיפה |
| **Limelight** | חיבור לליימלייט (team number / USB / limelight.local), סטטוס חי, וידאו, MegaTag + MegaTag2 על המגרש, **העלאת כיול**, גיבוי/שחזור, העלאת field map |

## כיול ליימלייט לתחרות (בלי שהאפליקציה פתוחה בזמן משחק)
1. בממשק של הליימלייט: pipeline בלי overlays (Viewfinder) ברזולוציה של התחרות (LL3: ‏1280x960, ‏LL3G/LL4: ‏1280x800).
2. Limelight → **Connect** → **Use Limelight stream as the app camera** – הסטרים של הליימלייט הופך למצלמה של האפליקציה.
3. Calibration → מדפיסים לוח, מצלמים 25–50 תמונות מגוונות → **Calibrate** (‏RMS < 1, עדיף 0.3–0.8).
4. Limelight → **Calibration** → **Upload calibration to Limelight**:
   האפליקציה קוראת את הכיול הקיים מהליימלייט, **מגבה** אותו ל‑`limelight_backups/`, מראה בדיוק אילו ערכים ישתנו, מעלה (`/cal-file`) ומוודאת בקריאה חוזרת.
5. זהו. הכיול נשמר על הליימלייט, שורד הפעלה מחדש, ומשמש בכל pipeline של AprilTags – בלי מחשב.

שחזור: **Restore backup…** או **Remove custom calibration** (חזרה לכיול המפעל).

## כיול אור אוטומטי לליימלייטים (Limelight → Auto light)
מוצא לכל ליימלייט את **החשיפה הקצרה ביותר** שבה כל התגים בתמונה מזוהים כמעט בכל פריים עם רעש נמוך (= הכי מעט motion blur בזמן נסיעה), ומוסיף מרווח ביטחון.

1. על המגרש האמיתי (זמן field calibration / practice): הרובוט עומד, התגים בטווח של משחק.
2. מוסיפים את הליימלייטים לרשימה (‏+ connected / ‏+ team .11 .12 .13 / כתובת ידנית) → **Auto-tune lighting**.
3. לכל ליימלייט: סריקת חשיפות (‏0.1–10ms), עידון סביב הסף, אימות – ושמירה על הליימלייט (`/update-pipeline?flush=1`).
4. אפשר לשמור **פרופיל תאורה** לכל אירוע/מגרש ולהחיל אותו בלחיצה על כל הליימלייטים.

בטיחות: בזמן הסריקה שום דבר לא נכתב לדיסק של הליימלייט; Stop / שגיאה / ריסטארט מחזירים את המקור. **Revert selected** מחזיר את הערכים שהיו לפני הכיול.

## כיול אור תוך כדי משחק (קוד רובוט)
המחשב לא מחובר בזמן משחק, אז החלק הזה רץ **בקוד הרובוט** (Java / WPILib):
1. Limelight → Auto light → **During the match** → **Fill from tune results** (או Add row), לבדוק IP סטטי ו‑NT name.
2. **Export Java file…** → לבחור את `src/main/java/frc/robot/vision` בפרויקט הרובוט. נוצר `LimelightAdaptiveExposure.java` (ה‑package מזוהה אוטומטית) וקוד השימוש מועתק ללוח.
3. ב‑`robotPeriodic()`: ‏`update(pose, robotRelativeSpeeds)`; ב‑`autonomousInit()`: ‏`resetToBaseline()`; אופציונלי `setHold(true)` בזמן כיוון/ירי.

מה הקוד עושה: לפי מיקום הרובוט הוא מחשב אילו תגים **אמורים** להיראות, משווה למה שהליימלייט מזהה (`rawfiducials` + ambiguity), ואם הזיהוי יורד – משנה חשיפה בצעדים (×1.2) לכיוון שמשפר (perturb & observe): כהה → מעלה, רוויה/טשטוש בנסיעה מהירה → מוריד.
מוגבל לטווח סביב הערך מהכיול, שינוי אחד לכל ~0.75 שנ', לא נשמר לדיסק (כל משחק מתחיל מהבסיס), נעצר אחרי 5 כשלי HTTP, ו‑`AdaptiveExposure/<name>/enabled` ב‑NetworkTables הוא מתג כיבוי. טלמטריה ב‑AdvantageScope / Elastic.

## מצלמת USB על הרובוריו (roboRIO camera, ‏Ctrl+5)
שליטה במצלמה שמחוברת ל‑USB של הרובוריו **בלי לשנות קוד רובוט**: WPILib CameraServer מריץ על הרובוריו שרת MJPEG (cscore, פורט 1181+) שמקבל שינויי הגדרות ישירות ב‑HTTP (`/?action=command&brightness=60`, `/settings.json`). קוד הרובוט לא רואה את השינויים.
- התנאי היחיד: בקוד הרובוט קיימת השורה `CameraServer.startAutomaticCapture();` (רוב הקבוצות כבר עם זה).
- **Find cameras** → כל ההגדרות שהמצלמה תומכת בהן (חשיפה, gain, בהירות, white balance, פוקוס…) נבנות אוטומטית.
- Presets: ‏Auto / Fast motion (פחות טשטוש בנסיעה) / Low light.
- Stream: רזולוציה / fps / איכות JPEG רק לצפייה באפליקציה – חוסך רוחב פס (מגבלת 4 Mbit/s במגרש).
- Profiles + **Keep applied**: אחרי ריסטארט של הרובוט המצלמה חוזרת לברירת מחדל – האפליקציה מחזירה את הפרופיל אוטומטית (גם מה‑Driver Station בזמן משחק – פורטים 1180–1190 פתוחים ברשת השדה).

## מפות מגרש
`fields/` – ‏2026 REBUILT ‏(Welded / AndyMark) ו‑2025 Reefscape (נתוני WPILib, ‏origin בפינת הברית הכחולה).
אפשר לטעון WPILib ‏`.json` או Limelight ‏`.fmap`, ולייצא/להעלות ‎.fmap לליימלייט (לליימלייט כבר יש את המפה הרשמית – נדרש רק למגרש אימון / מעבר Welded↔AndyMark).

## תיקיות שנוצרות
- `calibrations/` – כיולים של האפליקציה (נטענים אוטומטית לפי מצלמה + רזולוציה)
- `limelight_backups/` – גיבויים של כיולי ליימלייט לפני כל שינוי
- `recordings/` – הקלטות CSV
- `lighting_profiles.json` – פרופילי תאורה לליימלייטים
