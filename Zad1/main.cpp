#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <functional>
#include <string>
#include <iomanip>

using namespace std;

class Plansza;

struct Statystyki {
    int odwiedzone;
    int przetworzone;
    int maks_glebokosc;
    double czas_wykonania;
}; //Statystyki

Plansza* bfs(const Plansza& startowa, const string& kolejnosc, Statystyki& stats);
Plansza* dfs(const Plansza& startowa, const string& kolejnosc, int maks_glebokosc, Statystyki& stats);
Plansza* a_star(const Plansza& startowa, function<int(const Plansza&)> heurystyka, Statystyki& stats);
void zapisz_rozwiazanie(const string& nazwa_pliku, Plansza* rozwiazanie);
void zapisz_statystyki(const string& nazwa_pliku, Plansza* rozwiazanie, int odwiedzone, int przetworzone, int maks_glebokosc, double czas_wykonania);

class Plansza {
public:
    int wiersze;
    int kolumny;
    vector<vector<int>> plansza;
    pair<int, int> puste_pole;
    string sciezka;

    Plansza(const string& nazwa_pliku) {
        ifstream plik(nazwa_pliku);
        if (!plik) {
            cerr << "Nie mozna otworzyc pliku: " << nazwa_pliku << endl;
            exit(1);
        }

        plik >> wiersze >> kolumny;
        plansza.resize(wiersze, vector<int>(kolumny));

        for (int i = 0; i < wiersze; i++) {
            for (int j = 0; j < kolumny; j++) {
                plik >> plansza[i][j];
                if (plansza[i][j] == 0) {
                    puste_pole = {i, j};
                }
            }
        }
    } // Konstruktor

    Plansza(const Plansza& other) {
        wiersze = other.wiersze;
        kolumny = other.kolumny;
        plansza = other.plansza;
        puste_pole = other.puste_pole;
        sciezka = other.sciezka;
    } // Konstruktor kopiujacy

    bool operator==(const Plansza& other) const {
        return plansza == other.plansza;
    } // Operator porownania


    bool czy_rozwiazana() const {
        int k = 1;
        for (int i = 0; i < wiersze; i++) {
            for (int j = 0; j < kolumny; j++) {
                if (i == wiersze-1 && j == kolumny-1) {
                    if (plansza[i][j] != 0) return false;
                } else {
                    if (plansza[i][j] != k) return false;
                    k++;
                }
            }
        }
        return true;
    } // czy rozwiazana
    bool ruch_w_lewo() {
        int i = puste_pole.first;
        int j = puste_pole.second;
        if (j == 0) return false;

        swap(plansza[i][j], plansza[i][j-1]);
        puste_pole = {i, j-1};
        sciezka += "L";
        return true;
    }//lewo

    bool ruch_w_prawo() {
        int i = puste_pole.first;
        int j = puste_pole.second;
        if (j == kolumny - 1) return false;

        swap(plansza[i][j], plansza[i][j+1]);
        puste_pole = {i, j+1};
        sciezka += "R";
        return true;
    }//prawo

    bool ruch_w_gore() {
        int i = puste_pole.first;
        int j = puste_pole.second;
        if (i == 0) return false;

        swap(plansza[i][j], plansza[i-1][j]);
        puste_pole = {i-1, j};
        sciezka += "U";
        return true;
    }//gora

    bool ruch_w_dol() {
        int i = puste_pole.first;
        int j = puste_pole.second;
        if (i == wiersze - 1) return false;

        swap(plansza[i][j], plansza[i+1][j]);
        puste_pole = {i+1, j};
        sciezka += "D";
        return true;
    }//dol

    int hamming() const {
        int odleglosc = 0;
        int k = 1;
        for (int i = 0; i < wiersze; i++) {
            for (int j = 0; j < kolumny; j++) {
                if (i == wiersze-1 && j == kolumny-1) {
                    if (plansza[i][j] != 0) odleglosc++;
                } else {
                    if (plansza[i][j] != k) odleglosc++;
                    k++;
                }
            }
        }
        return odleglosc;
    } //haming

    int manhattan() const {
        int odleglosc = 0;
        for (int i = 0; i < wiersze; i++) {
            for (int j = 0; j < kolumny; j++) {
                if (plansza[i][j] == 0) continue;

                int prawidlowe_i = (plansza[i][j] - 1) / kolumny;
                int prawidlowe_j = (plansza[i][j] - 1) % kolumny;
                odleglosc += abs(i - prawidlowe_i) + abs(j - prawidlowe_j);
            }
        }
        return odleglosc;
    } //manhattan

    vector<Plansza> generuj_sasiadow(const string& kolejnosc) const {
        vector<Plansza> sasiedzi;
        for (char kierunek : kolejnosc) {
            Plansza kopia(*this);
            bool udany_ruch = false;

            switch (kierunek) {
                case 'L': udany_ruch = kopia.ruch_w_lewo(); break;
                case 'R': udany_ruch = kopia.ruch_w_prawo(); break;
                case 'U': udany_ruch = kopia.ruch_w_gore(); break;
                case 'D': udany_ruch = kopia.ruch_w_dol(); break;
            }

            if (udany_ruch) {
                sasiedzi.push_back(kopia);
            }
        }
        return sasiedzi;
    } //generowanie sasiadow

    string znajdz_sciezke() const {
        return sciezka;
    } //znajdz sciezke

    string plansza_string() const {
        stringstream ss;
        for (const auto& wiersz : plansza) {
            for (int pole : wiersz) {
                ss << pole << " ";
            }
        }
        return ss.str();
    } //plansza string
}; //Plansza


struct AStarComparator {
    function<int(const Plansza&)> heurystyka;

    AStarComparator(function<int(const Plansza&)> h) : heurystyka(h) {}

    bool operator()(const Plansza& a, const Plansza& b) const {
        return a.sciezka.length() + heurystyka(a) > b.sciezka.length() + heurystyka(b);
    }
}; //A* z heurystyka

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Error: Nieprawidlowa liczba argumentow\n";
        return 1;
    }

    string strategia = argv[1];
    string parametr = argv[2];
    string plik_wejsciowy = argv[3];
    string plik_rozwiazanie = argv[4];
    string plik_statystyki = argv[5];

    Plansza startowa(plik_wejsciowy);
    Plansza* rozwiazanie = nullptr;
    Statystyki stats;

    if (strategia == "bfs") {
        rozwiazanie = bfs(startowa, parametr, stats);
    }
    else if (strategia == "dfs") {
        int maks_glebokosc = 20;
        rozwiazanie = dfs(startowa, parametr, maks_glebokosc, stats);
    }
    else if (strategia == "astr") {
        function<int(const Plansza&)> heurystyka;
        if (parametr == "hamm") {
            heurystyka = [](const Plansza& p) { return p.hamming(); };
        } else if (parametr == "manh") {
            heurystyka = [](const Plansza& p) { return p.manhattan(); };
        } else {
            cerr << "Nieznana heurystyka: " << parametr << endl;
            return 1;
        }
        rozwiazanie = a_star(startowa, heurystyka, stats);
    }
    else {
        cerr << "Nieznana strategia: " << strategia << endl;
        return 1;
    }

    zapisz_rozwiazanie(plik_rozwiazanie, rozwiazanie);
    zapisz_statystyki(plik_statystyki, rozwiazanie, stats.odwiedzone, stats.przetworzone, stats.maks_glebokosc, stats.czas_wykonania);

    if (rozwiazanie != nullptr) {
        delete rozwiazanie;
    }

    return 0;
} //main

Plansza* bfs(const Plansza& startowa, const string& kolejnosc, Statystyki& stats) {
    auto start_time = chrono::high_resolution_clock::now();

    queue<Plansza> kolejka;
    unordered_set<string> odwiedzone;

    kolejka.push(startowa);
    odwiedzone.insert(startowa.plansza_string());
    stats.odwiedzone = 1;
    stats.przetworzone = 0;
    stats.maks_glebokosc = 0;

    while (!kolejka.empty()) {
        Plansza obecna = kolejka.front();
        kolejka.pop();
        stats.przetworzone++;

        if (obecna.czy_rozwiazana()) {
            stats.czas_wykonania = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            return new Plansza(obecna);
        }

        vector<Plansza> sasiedzi = obecna.generuj_sasiadow(kolejnosc);
        for (Plansza& sasiad : sasiedzi) {
            string sasiad_str = sasiad.plansza_string();
            if (odwiedzone.find(sasiad_str) == odwiedzone.end()) {
                odwiedzone.insert(sasiad_str);
                kolejka.push(sasiad);
                stats.odwiedzone++;

                int glebokosc = sasiad.sciezka.length();
                if (glebokosc > stats.maks_glebokosc) {
                    stats.maks_glebokosc = glebokosc;
                }
            }
        }
    }

    stats.czas_wykonania = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    return nullptr;
} //BFS


Plansza* dfs(const Plansza& startowa, const string& kolejnosc, int maks_glebokosc, Statystyki& stats) {
    auto start_time = chrono::high_resolution_clock::now();

    stack<Plansza> stos;
    unordered_set<string> odwiedzone;

    stos.push(startowa);
    odwiedzone.insert(startowa.plansza_string());
    stats.odwiedzone = 1;
    stats.przetworzone = 0;
    stats.maks_glebokosc = 0;

    while (!stos.empty()) {
        Plansza obecna = stos.top();
        stos.pop();
        stats.przetworzone++;

        if (obecna.czy_rozwiazana()) {
            stats.czas_wykonania = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            return new Plansza(obecna);
        }

        if (obecna.sciezka.length() >= maks_glebokosc) {
            continue;
        }

        vector<Plansza> sasiedzi = obecna.generuj_sasiadow(kolejnosc);
        for (Plansza& sasiad : sasiedzi) {
            string sasiad_str = sasiad.plansza_string();
            if (odwiedzone.find(sasiad_str) == odwiedzone.end()) {
                odwiedzone.insert(sasiad_str);
                stos.push(sasiad);
                stats.odwiedzone++;

                int glebokosc = sasiad.sciezka.length();
                if (glebokosc > stats.maks_glebokosc) {
                    stats.maks_glebokosc = glebokosc;
                }
            }
        }
    }

    stats.czas_wykonania = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    return nullptr;
} //DFS

Plansza* a_star(const Plansza& startowa, function<int(const Plansza&)> heurystyka, Statystyki& stats) {
    auto start_time = chrono::high_resolution_clock::now();

    priority_queue<Plansza, vector<Plansza>, AStarComparator> kolejka((AStarComparator(heurystyka)));

    unordered_set<string> odwiedzone;

    kolejka.push(startowa);
    odwiedzone.insert(startowa.plansza_string());
    stats.odwiedzone = 1;
    stats.przetworzone = 0;
    stats.maks_glebokosc = 0;

    while (!kolejka.empty()) {
        Plansza obecna = kolejka.top();
        kolejka.pop();
        stats.przetworzone++;

        if (obecna.czy_rozwiazana()) {
            stats.czas_wykonania = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            return new Plansza(obecna);
        }

        vector<Plansza> sasiedzi = obecna.generuj_sasiadow("LRUD");
        for (Plansza& sasiad : sasiedzi){
            string sasiad_str = sasiad.plansza_string();
            if (odwiedzone.find(sasiad_str) == odwiedzone.end()) {
                odwiedzone.insert(sasiad_str);
                kolejka.push(sasiad);
                stats.odwiedzone++;

                int glebokosc =sasiad.sciezka.length();
                if (glebokosc > stats.maks_glebokosc) {
                    stats.maks_glebokosc = glebokosc;
                }
            }
        }
    }

    stats.czas_wykonania = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    return nullptr;
} //a_star

void zapisz_rozwiazanie(const string& nazwa_pliku, Plansza* rozwiazanie) {
    ofstream plik(nazwa_pliku);
    if (!plik) {
        cerr << "Nie mozna otworzyc pliku: " << nazwa_pliku << endl;
        exit(1);
    }

    if (rozwiazanie == nullptr) {
        plik << "-1\n";
    } else {
        plik << rozwiazanie->sciezka.length() << "\n";
        plik << rozwiazanie->sciezka << "\n";
    }
} //zapisywanie rozw

void zapisz_statystyki(const string& nazwa_pliku, Plansza* rozwiazanie, int odwiedzone, int przetworzone, int maks_glebokosc, double czas_wykonania) {
    ofstream plik(nazwa_pliku);
    if (!plik) {
        cerr << "Nie mozna otworzyc pliku: " << nazwa_pliku << endl;
        exit(1);
    }

    if (rozwiazanie == nullptr) {
        plik << "-1\n";
    } else {
        plik << rozwiazanie->sciezka.length() << "\n";
    }
    plik << odwiedzone << "\n";
    plik << przetworzone << "\n";
    plik << maks_glebokosc << "\n";
    plik << fixed << setprecision(15) << czas_wykonania << "\n";
} //zapisywanie stats
