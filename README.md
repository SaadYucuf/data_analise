# Mijozlarning Ketishini Bashorat Qilish (Telco Customer Churn Prediction)

![Project Banner](https://img.freepik.com/free-vector/customer-feedback-user-experience-flat-design_23-2148943764.jpg?size=626&ext=jpg)

Ushbu loyiha Telekom kompaniyasi mijozlarining ketib qolish (churn) holatlarini tahlil qilish va bashorat qilishga qaratilgan. Loyihaning asosiy maqsadi – ma'lumotlarga asoslangan holda biznes qarorlarini qabul qilishni qo'llab-quvvatlash va mijozlarni saqlab qolish strategiyalarini ishlab chiqish uchun chuqur tahlillar taqdim etish.

## Loyihaning Asosiy Maqsadlari

- **Tahlil (Analysis):** Mijozlar ketishiga ta'sir qiluvchi asosiy omillarni (key drivers) aniqlash.
- **Bashorat (Prediction):** Qaysi mijozlar yaqin kelajakda kompaniyadan ketish ehtimoli yuqori ekanligini Machine Learning yordamida oldindan aytib berish.
- **Hisobot (Reporting):** Barcha topilmalar, grafiklar va tavsiyalarni avtomatik tarzda professional MS Word (.docx) hisobotiga jamlash.

---

## 🚀 Texnologiyalar Steki

- **Dasturlash tili:** Python 3
- **Asosiy kutubxonalar:**
  - **Ma'lumotlar bilan ishlash:** Pandas, NumPy
  - **Vizualizatsiya:** Matplotlib, Seaborn
  - **Machine Learning:** Scikit-learn, XGBoost
  - **Hisobot generatsiyasi:** python-docx
  - **Ma'lumotlar manbasi:** KaggleHub API

---

## 📊 Asosiy Topilmalar va Grafiklar

Loyihada bir nechta muhim biznes-tushunchalar aniqlandi. Quyida ulardan ba'zilari keltirilgan:

### 1. Muammoning Ko'lami
Kompaniya mijozlarining **26.5%**ini yo'qotmoqda. Bu biznes uchun barqaror o'sishga to'sqinlik qiladigan jiddiy ko'rsatkich.
![Umumiy Churn](httpss://user-images.githubusercontent.com/username/project/churn_distribution.png)  <!-- Bu yerga o'z grafik rasmingiz havolasini qo'yasiz -->

### 2. Asosiy Sabablar
- **Qisqa muddatli shartnomalar (`Month-to-month`):** Mijozlar ketishining eng asosiy sababchisi.
- **"Fiber Optic" xizmati:** Ajablanarlisi, ushbu premium xizmatdan foydalanuvchilar orasida ketish darajasi yuqori, bu esa narx yoki sifat muammosidan dalolat berishi mumkin.
![Contract Churn](https://user-images.githubusercontent.com/username/project/contract_churn.png) <!-- Bu yerga o'z grafik rasmingiz havolasini qo'yasiz -->

### 3. Model Natijalari
- **Eng yaxshi model:** `XGBoost Classifier`.
- **Sifat ko'rsatkichi (AUC):** **0.85**, bu "juda yaxshi" natija hisoblanadi.
- **Biznes qiymati:** Modelimiz ketishi mumkin bo'lgan har 100 mijozdan **66 tasini** to'g'ri aniqlay oladi (Recall=0.66). Bu esa marketing harakatlarini aniq yo'naltirish imkonini beradi.
![ROC AUC Curve](https://user-images.githubusercontent.com/username/project/roc_auc.png) <!-- Bu yerga o'z grafik rasmingiz havolasini qo'yasiz -->

---

## 🛠️ Loyihani Ishga Tushirish

Loyihani o'z kompyuteringizda ishga tushirish uchun quyidagi qadamlarni bajaring:

**1. Repozitoriyni klonlash:**
```bash
git clone https://github.com/your-username/telco-churn-analysis.git
cd telco-churn-analysis
```

**2. Virtual muhit yaratish (Tavsiya etiladi):**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux uchun
# venv\Scripts\activate  # Windows uchun
```

**3. Kerakli kutubxonalarni o'rnatish:**
```bash
pip install -r requirements.txt
```
**Eslatma:** Kaggle'dan ma'lumot yuklash uchun `kaggle.json` faylingiz to'g'ri sozlangan bo'lishi kerak.

**4. Dasturni ishga tushirish:**
```bash
python main_script.py  # Asosiy skriptingiz nomini yozing
```

Dastur ishga tushgandan so'ng, barcha tahlillar va grafiklar ekranda paydo bo'ladi. Eng so'ngida, loyiha papkasida `Mijozlar_Ketishi_Tahlili_Hisoboti.docx` nomli yakuniy Word hisoboti yaratiladi.

---

## ✒️ Muallif

- **[Sizning Ismingiz]**
- **Portfolio:** [Portfolio saytingizga havola (agar mavjud bo'lsa)]
- **LinkedIn:** [LinkedIn profilingizga havola]
- **GitHub:** [@your-github-username]
