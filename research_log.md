Tanggal:
9 Juni 2026

Tujuan:
Menguji kemampuan Whisper Tiny mengenali huruf hijaiyah.

Model:
Whisper Tiny

Audio Input:
"a ba ta tsa a ba ta a"

Output:
"أبك أرساء"

Observasi:
Model gagal mengenali huruf hijaiyah terisolasi.
Model menghasilkan kata Arab yang memiliki kemiripan fonetik.

Kesimpulan:
Whisper kurang cocok untuk pengenalan huruf tunggal tanpa fine-tuning.

Tanggal:
10 Juni 2026

Tujuan:
Menguji pengenalan frasa Arab sederhana.

Input:
السلام عليكم

Output:
السلام وانيكم

Observasi:
Kata pertama dikenali dengan benar.
Kata kedua mengalami kesalahan transkripsi.

Kesimpulan:
Whisper menunjukkan performa lebih baik pada frasa dibanding huruf tunggal.

Tanggal:
10 Juni 2026

Tujuan:
Menguji pengenalan ayat pendek.

Input:
بسم الله الرحمن الرحيم

Output:
بس من لا يرواح ما نرى حيم

Observasi:
Banyak kesalahan pada kata-kata Quran.
Model cenderung mengganti dengan kata Arab lain yang mirip bunyi.

Kesimpulan:
Whisper standar belum optimal untuk bacaan Quran.

Observasi:

Whisper Tiny gagal mengenali seluruh huruf hijaiyah berharakat fathah yang diuji.

Model tidak menghasilkan huruf tunggal seperti اَ, بَ, تَ, ثَ, dan جَ,
melainkan menghasilkan kata-kata Arab yang memiliki kemiripan fonetik.

Contohnya:

اَ → أه
بَ → باب
ثَ → شاء
جَ → جاء

Hal ini menunjukkan bahwa Whisper cenderung menafsirkan audio sebagai kata atau frasa Arab yang bermakna, bukan sebagai huruf hijaiyah tunggal.

Akurasi baseline yang diperoleh adalah 0%.

Observasi:

Whisper Base belum mampu mengenali huruf hijaiyah berharakat fathah secara langsung.

Model tidak menghasilkan transkripsi berupa huruf tunggal seperti اَ, بَ, تَ, ثَ, dan جَ, melainkan menghasilkan kata atau frasa dalam bahasa Arab yang memiliki kemiripan fonetik dengan bunyi yang diucapkan.

Contohnya:

اَ → أه
بَ → با / بات
تَ → تأكد / حسنة / لا
ثَ → فرق / ها
جَ → جاء / دعو

Hasil ini menunjukkan bahwa meskipun Whisper Base mampu menangkap karakteristik bunyi bahasa Arab, model tetap menginterpretasikan audio sebagai kata atau kalimat yang bermakna, bukan sebagai huruf hijaiyah tunggal.

Pada pengujian terhadap 20 data uji, model tidak berhasil mengenali satu pun huruf hijaiyah dengan benar sehingga memperoleh baseline accuracy sebesar 0%.