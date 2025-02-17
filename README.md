# Reinforcement Learning ile Fetch Robot Eğitimi

Bu proje, gerçek dünyada var olan Fetch robotunu, çeşitli Takviyeli Öğrenme (Reinforcement Learning - RL) algoritmalarıyla eğitmeyi ve farklı ortamlarda nasıl performans gösterdiğini analiz etmeyi amaçlamaktadır.
Proje Açıklaması

Bu çalışma kapsamında, OpenAI Gym'in güncel versiyonu olan Gymnasium Farama ve Stable-Baselines3 kütüphanesi kullanılarak FetchReach, FetchPush, FetchPickAndPlace ve FetchSlide ortamlarında robot eğitimi gerçekleştirilmiştir. Eğitim sürecinde PPO, DDPG, SAC, TQC ve TD3 algoritmaları kullanılarak her ortamın kendine özgü hedeflerini tamamlayabilmesi için detaylı analizler yapılmıştır.
Kullanılan Teknolojiler

    Python 3.8+
    Gymnasium (Farama)
    MuJoCo (Robot simülasyonu için)
    Stable-Baselines3
    TensorFlow & PyTorch (Takviyeli öğrenme modelleri için)
    TensorBoard (Eğitim sürecinin görselleştirilmesi için)

# MuJoCo'yu yüklemek için:

pip install mujoco

# Stable-Baselines3 ve Gymnasium için:

pip install stable-baselines3 gymnasium[robotics]

# Kullanım

Proje, dört farklı ortamda Fetch robotunun eğitimini ve RL algoritmalarının performansını test etmek için oluşturulmuştur.

Eğitimi başlatmak için:

python train.py --env FetchPush-v2 --algo SAC --timesteps 1000000

Eğitilmiş modeli test etmek için:

python test.py --model trained_models/sac_fetchpush.zip

# Deneyler ve Sonuçlar

    FetchReach ortamında tüm algoritmalar başarılı olurken, SAC en hızlı öğrenen algoritma olmuştur.
    FetchPush ve FetchPickAndPlace ortamlarında TQC ve SAC en yüksek başarı oranını elde etmiştir.
    FetchSlide ortamı, yüksek karmaşıklığından dolayı yalnızca TQC algoritması ile kısmen başarılı sonuçlar vermiştir.

# Parametre Tuning Çalışmaları

Projede hyperparameter tuning için özel bir program geliştirilmiş ve batch size, learning rate, entropy coefficient gibi parametrelerin performansa etkisi analiz edilmiştir.
# Geliştirme Süreci

    Model eğitimi ve analizi
    Performans kıyaslamaları ve grafik oluşturma
    Hiperparametre optimizasyonu
    Simülasyon verilerinin analizi
