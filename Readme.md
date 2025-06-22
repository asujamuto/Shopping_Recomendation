Nie wrzuciłem folderu data, w którym są rozpakowane dane z kaggle.
Do zainstalowania: pandas, numpy, jupyter, ipykernel

### Dashboard Gradio v1
`title + store + description` jest wrzucane do tf-ifd matrix

Uwagi 
- w datasecie treningowym może być mało interakcji dla niektórych kategorii(np. nie ma użytkowniów z przynajmniej 4 interakcjami którzy mieliby przynajmniej 2 interakcje z Home Audio)
- w przypadku małej liczby poprzednich interakcji rekomendowane są różne wersje tych samych produktów (np. mahjongm TurboTax)
- rekomendowane są te same produkty

Todo:
- [ ] caching wyszukiwań do pliku (albo precompute wszystkiego na raz)
- [ ] opcja wyboru użytkowników spośród znalezionych bez powtarzania wyszkuniwania (i tak otrzymujemy całą listę użytkowników dla danego wyszukiwania)