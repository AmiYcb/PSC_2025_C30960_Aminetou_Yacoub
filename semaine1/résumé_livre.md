# Résumé  du livre

## 1. Démarrage avec Python et préparation de l’environnement

### Installation recommandée
 
 Le livre suggère d’utiliser Anaconda, une distribution Python complète, idéale pour débuter car elle :

simplifie la gestion des environnements virtuels,

inclut Python, IPython, Jupyter Notebook, NumPy, matplotlib, SciPy, et l’éditeur Spyder.

Cela évite de modifier l’installation système de Python et permet une désinstallation propre en supprimant simplement le dossier Anaconda.

Version de Python
Les exemples utilisent Python 3.x, à partir de la version 3.7.

Spyder : l’IDE scientifique
Spyder est un environnement de développement interactif qui présente trois panneaux par défaut :

un éditeur de code,

une console IPython,

un panneau d’aide.

On recommande de débuter chaque script avec :

```python
from numpy import *
from matplotlib.pyplot import *
```

pour disposer des fonctions principales.

Spyder affiche aussi des avertissements (triangle jaune) pour certains usages comme `from numpy import *`, mais ces avertissements sont volontairement ignorés ici.

### IPython et Jupyter

- **IPython** remplace avantageusement le shell Python standard, offrant :
    - aide rapide (`?`),
    - complétion (`Tab`),
    - historique,
    - commandes magiques (`%magic`, `%timeit`, `%run`),
    - sortie formatée.
- **Jupyter Notebook** permet de créer des documents interactifs mêlant texte, code, graphiques, idéal pour :
    - les devoirs,
    - les notes de cours,
    - les démonstrations.

Pour lancer Jupyter :

```bash
jupyter notebook

```

Un navigateur s’ouvre alors, et on peut interagir avec Python via son interface web.

### Exécution de scripts

Dans IPython :

- `cd` pour naviguer dans les dossiers.
- `run myscript.py` pour exécuter le script.

---

## 2. Variables et types de données fondamentaux

### Variables

- Pas de déclaration explicite du type, Python déduit automatiquement.
- Les noms de variables peuvent contenir des lettres, chiffres et underscores, mais ne commencent jamais par un chiffre.
- La casse est significative (`variable` ≠ `Variable`).

### Types numériques

### Entiers

- `int` représente l’ensemble des entiers relatifs.
- Non borné, limité uniquement par la mémoire.

### Flottants

- `float` : sous-ensemble fini des réels.
- Exemple :

```python
6 // 2 # donne 3 (int)
7 // 2 # donne 3
7 / 2  # donne 3.5 (float)

```

- Attention aux erreurs d’arrondi :

```python

0.4 - 0.3 # donne 0.10000000000000003

```

- `sys.float_info.epsilon` ≈ `2.220446049250313e-16`.

### Complexes

- Python utilise `j` pour l’unité imaginaire.

```python

z = 3.5 + 5.2j
z.real # 3.5
z.imag # 5.2
z.conjugate() # (3.5 - 5.2j)

```

### Infini et NaN

- `inf` représente l’infini, `nan` signifie « not-a-number ».

```python
from numpy import *
a = exp(1000) # inf
a - a # nan

```

- Comparer `nan` avec quoi que ce soit retourne toujours `False`.

### Booléens

- `True` et `False` sont en réalité des entiers (`int`), hérités par sous-classement.
- Les opérateurs logiques sont `and`, `or`, `not`.
- Les comparaisons peuvent être chaînées :

```python

2 < 3 < 4 # True

```

- Les objets « vides » (0, `[]`, `''`) se convertissent en `False`.

### Chaînes de caractères

- Définies par `'...'` ou `"..."`, multi-lignes avec `"""..."""`.
- Échappements :
    
    Python utilise plusieurs séquences d’échappement pour représenter des caractères spéciaux :
    
    - `\n` pour une nouvelle ligne (newline)
    - `\t` pour une tabulation horizontale (tab)
    - `\\` pour un anti-slash littéral
    - `\"` pour un guillemet double dans une chaîne délimitée par `"`
- Raw strings : `r"\t"`.

### Opérations

- Concaténation : `'a' + 'b'`
- Répétition : `'Hi'*3` → `'HiHiHi'`
- Comparaison lexicographique : `'Anna' < 'Arvi'`

### Méthodes courantes

- `.split()`, `.join()`, `.find()`.
- Formatage :

```python
 
f"Le cours {course} a pour note {note:.2f}"
"{} {}".format('Hello', 'World')

```

---

# 3. Conteneurs standards en Python : listes, tuples, dictionnaires, ensembles et arrays

## Introduction aux conteneurs

Python offre plusieurs types de **structures de données intégrées**, appelées **containers**, qui permettent de stocker et d’organiser des collections de valeurs.

Ces containers sont adaptés à différents usages :

- **listes** : séquences ordonnées, mutables, contenant tout type d’objet.
- **tuples** : séquences ordonnées, immuables.
- **dictionnaires** : ensembles non ordonnés de paires clé-valeur.
- **ensembles (sets)** : collections non ordonnées d’éléments uniques.
- **arrays NumPy** : tableaux multidimensionnels pour calculs numériques rapides.

---

## Listes

### Définition et indexation

Une liste est une séquence ordonnée et mutable.

Elle se définit avec des crochets `[]` :

```python
 L = [10, 20, 30, 40]
```

- Accès à un élément par son index (0-based) :

```python
L[0]  # 10
L[2]  # 30

```

- Index négatifs pour compter à rebours :

```python
python
 L[-1]  # 40

```

### Slicing

- Pour extraire une sous-liste :

```python
 L[1:3]  # [20, 30]

```

- Avec un pas (stride) :

```python
 L[::2]  # [10, 30]

```

- Les slices ne provoquent jamais d’erreur si les bornes dépassent.

### Modifications et méthodes

- On peut modifier une valeur par affectation directe :

```python
 L[1] = 99  # [10, 99, 30, 40]

```

- Les principales méthodes :

```python
 L.append(50)     # ajoute à la fin
L.extend([60,70]) # ajoute plusieurs
L.insert(2, 25)  # insère à l’index 2
L.remove(30)     # supprime la première occurrence
x = L.pop()      # retire et retourne le dernier élément
L.sort()         # trie en place
L.reverse()      # inverse l’ordre

```

### List comprehensions

Python offre une syntaxe compacte pour construire des listes :

```python
 [x**2 for x in range(5) if x % 2 == 0]  # [0, 4, 16]

```

Cela remplace avantageusement une boucle `for` classique.

---

## Tuples

### Définition

Les tuples ressemblent aux listes mais sont **immuables**.

Ils se créent avec des parenthèses :

```python
 t = (1, 2, 3)

```

- Accès identique aux listes :

```python
 t[0]  # 1

```

- Utilisés souvent pour retourner plusieurs valeurs :

```python
 def compute():
    return 4, 5
a, b = compute()  # unpacking

```

- Un tuple à un élément doit avoir une virgule :

```python
 single = (42,)

```

---

## Dictionnaires

### Définition

Les dictionnaires (`dict`) sont des **collections non ordonnées** associant des clés uniques à des valeurs.

```python
 d = {'name': 'Alice', 'age': 30}

```

- Accès via la clé :

```python
 d['name']  # 'Alice'

```

### Itérations

```python
 for k in d.keys():
    print(k)

for v in d.values():
    print(v)

for k, v in d.items():
    print(f"{k} -> {v}")

```

### Méthodes utiles

- `d.get('name', 'inconnu')` : retourne la valeur ou 'inconnu' si clé absente.
- `d.pop('age')` supprime la clé et retourne la valeur.

---

## Ensembles (sets)

### Définition

Un ensemble (`set`) est une collection d’éléments **uniques et non ordonnés**.

```python
 s = {1, 2, 3, 3}
print(s)  # {1, 2, 3}

```

- Ajouter ou retirer des éléments :

```python
 s.add(4)
s.remove(2)

```

### Opérations ensemblistes

Les sets permettent des opérations mathématiques rapides :

```python
 A = {1, 2, 3}
B = {3, 4, 5}
A.union(B)       # {1,2,3,4,5}
A.intersection(B) # {3}
A.difference(B)   # {1,2}

```

---

## Arrays NumPy

### Pourquoi NumPy

Les **arrays NumPy** sont spécialement conçus pour le calcul scientifique :

- stockage compact en mémoire,
- opérations vectorisées très rapides.

```python
 from numpy import array
A = array([[1, 2], [3, 4]])

```

- Indexation 2D :

```python
 A[1, 0]  # 3

```

- Opérations élément par élément :

```python
 A + A  # [[2,4],[6,8]]
A * A  # [[1,4],[9,16]]

```

---

## Conversion entre types

### Conversion standard

Python facilite la conversion entre containers :

```python
 list((1,2,3))  # [1,2,3]
set([1,1,2,3]) # {1,2,3}
tuple('abc')   # ('a', 'b', 'c')

```

Cela permet de passer d’une structure à l’autre selon les besoins (recherche rapide avec `set`, ordonnancement avec `list`, sécurité avec `tuple`).