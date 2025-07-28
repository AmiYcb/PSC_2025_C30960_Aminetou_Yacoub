
Chapitre 4 – Algèbre linéaire et tableaux NumPy
L'algèbre linéaire est un pilier des mathématiques computationnelles, traitant principalement des vecteurs et matrices. NumPy offre tous les outils nécessaires pour les manipuler efficacement.

4.1 Vue d’ensemble du type array
Cette section présente l’utilisation des tableaux NumPy, leur création et manipulation.

4.1.1 Vecteurs et Matrices
Importer NumPy : from numpy import *

Créer un vecteur : v = array([1., 2., 3.])

Opérations de base :

Multiplication/division scalaire, combinaisons linéaires, produit scalaire, produit matriciel, opérations élémentaires (v1 * v2, v1 + v2, cos(v1)), etc.

Exemple de produit scalaire : dot(v1, v2) ou v1 @ v2

Création de matrices :

```python
M = array([[1., 2.], [0., 1.]])
```
Conversion entre vecteur, matrice ligne, et matrice colonne :


```python
R = v.reshape((1, 3))  # Matrice ligne
C = v.reshape((3, 1))  # Matrice colonne
```
4.1.2 Indexation et slicing
Comme les listes Python, mais multidimensionnelles.

Exemples :

```python
v[0], v[1:], M[0, 0], M[1], M[1:]
v[0] = 10
v[:2] = [0, 1]
```
4.1.3 Opérations d’algèbre linéaire
Produits scalaires et matriciels : dot(M, v), dot(v, w), dot(M, N) ou avec @.

Résolution de systèmes linéaires :

```python
from numpy.linalg import solve
x = solve(A, b)
allclose(dot(A, x), b)
```
4.2 Préliminaires mathématiques
4.2.1 Les tableaux comme fonctions
Un vecteur est vu comme une fonction d’un indice vers une valeur.

Une matrice est une fonction de deux variables (ligne, colonne).

4.2.2 Opérations élémentaires
A * B est une multiplication élément par élément, pas un produit matriciel.

Toutes les opérations scalaires (+, -, *, /) sont élémentaires.

4.2.3 Dimensions et formes
Objet	Forme	Signification
Scalaire	()	Fonction sans argument
Vecteur	(n,)	Fonction avec un argument
Matrice	(m, n)	Fonction avec deux arguments
Tenseur	(d1, d2, ...)	Plus de deux dimensions

```python
shape(A), ndim(A)
```
4.2.4 Opérations de dot (produits)
Réduction structurée :

vecteur · vecteur → scalaire

matrice · vecteur → vecteur

matrice · matrice → matrice

Exemple :

```python
M = array([[cos(pi/3), -sin(pi/3)], [sin(pi/3), cos(pi/3)]])
v = array([1., 0.])
```

y = M @ v
4.3 Le type array
4.3.1 Propriétés
shape: taille

dtype: type des données

strides: sauts mémoire entre éléments

```python
A = array([[1, 2, 3], [3, 4, 6]])
```

A.shape, A.dtype, A.strides
4.3.2 Création à partir de listes
```python
array([1., 2., 3.], dtype=float)
```
Attention aux conversions implicites :

```python
a = array([1, 2, 3])
a[0] = 0.5  # devient 0 (tronqué)
```
4.4 Accès aux éléments
4.4.1 Indexation

```python
M[0, 0], M[-1, 0]
```
4.4.2 Slicing
```python
M[i, :], M[:, j], M[2:4, 1:4]
```

Les slices sont des vues, donc :

```python
v = array([1., 2., 3.])
v1 = v[:2]
v1[0] = 0.  # modifie aussi v
```

4.5 Fonctions pour construire des tableaux
Fonction	Description
zeros((n,m))	Matrice n×m remplie de zéros
ones((n,m))	Matrice remplie de uns
full((n,m), q)	Matrice remplie de q
diag(v)	Matrice diagonale
random.rand(n,m)	Valeurs aléatoires
arange(n)	Vecteur d’entiers de 0 à n-1
linspace(a,b,n)	n points entre a et b
identity(n)	Matrice identité

4.6 Accès et modification de forme
```python
v.reshape(2, 3)
v.reshape(2, -1)
```
A.T  # transpose
4.7 Empilement (stacking)
Fonction	Action
concatenate([...])	Concatenation le long d’un axe
hstack([...])	Empilement horizontal (colonne)
vstack([...])	Empilement vertical (ligne)
column_stack([...])	Empilement en colonnes

4.8 Fonctions appliquées aux arrays
4.8.1 Fonctions universelles (ufuncs)
```python
cos(array([0, pi/2, pi]))
array([1,2])**2
```
Créer une ufunc personnalisée :

```python
@vectorize
def heaviside(x):
    return 1. if x >= 0 else 0.
```
4.8.2 Fonctions globales sur tableaux
```python
sum(A), sum(A, axis=0), sum(A, axis=1)
```

4.9 Méthodes en algèbre linéaire (SciPy)
4.9.1 Factorisation LU
Pour résoudre efficacement plusieurs systèmes avec la même matrice A :

```python
from scipy.linalg import lu_factor, lu_solve
LU, piv = lu_factor(A)
x = lu_solve((LU, piv), b)
```
4.9.2 Moindres carrés avec SVD
```python
U, S, VT = scipy.linalg.svd(A, full_matrices=False)
x = VT.T @ ((U.T @ b) / S)
```

Autre solution directe :

```python
scipy.linalg.lstsq(A, b)
```

4.9.3 Autres fonctions utiles
Fonction	Rôle
det(A)	Déterminant
eig(A)	Valeurs/vecteurs propres
inv(A)	Inverse
pinv(A)	Pseudo-inverse
norm(A)	Norme
svd(A)	Décomposition SVD
lu(A)	Décomposition LU
qr(A)	Décomposition QR
cholesky(A)	Décomposition de Cholesky
solve(A, b)	Résolution de système linéaire
solve_banded(...)	Pour systèmes bande
lstsq(A, b)	Résolution moindres carrés

4.10 Résumé
Les tableaux NumPy représentent vecteurs, matrices et tenseurs.

L'algèbre linéaire est simplifiée grâce aux fonctions de numpy.linalg et surtout scipy.linalg pour plus de performance.

Cela pose les bases pour des manipulations plus complexes dans les chapitres suivants.


Chapitre 5 – Manipulation avancée des tableaux NumPy
5.1 Vues et copies de tableaux
Une vue est un objet léger qui fait référence aux mêmes données qu’un autre tableau (aucune copie de données).

```python

v = M[0, :]  # v est une vue de M
v[-1] = 0.   # modifie aussi M
```
Vérifier si un tableau est une vue : v.base est différent de None.

Seul le slicing basique (:) retourne une vue. Les indexations avancées retournent une copie.

Les méthodes .T (transposition) et .reshape() retournent souvent des vues.

Pour forcer une copie, utilisez array() :

```python
N = array(M.T)
N.base is None  # True → c’est une copie réelle
```
5.2 Comparaison de tableaux
5.2.1 Tableaux booléens
Les comparaisons produisent des tableaux booléens :

```python

A == B           # renvoie un tableau booléen
(A == B).all()   # renvoie un booléen True/False
```
5.2.2 Tester l’égalité
Pour comparer deux tableaux de flottants avec une certaine tolérance :

```python

allclose(A, B, rtol=1e-5, atol=1e-8)
Equivalent à :
(abs(A-B) < atol + rtol*abs(B)).all()
```
5.2.3 Opérateurs logiques booléens
Ne pas utiliser and, or, not pour les tableaux. Utiliser :

Logique	Utiliser à la place
and	&
or	`
not	~

Exemple :

```python
(deviation < -0.5) | (deviation > 0.5)
```
5.3 Indexation des tableaux
5.3.1 Indexation booléenne
Utilise un masque booléen :

```python
M = array([[2, 3], [1, 4]])
B = array([[True, False], [False, True]])
M[B] = [10, 20]  # modifie uniquement les éléments True
```
Utilisation conditionnelle :

```python
M[M > 2] = 0
```
5.3.2 Fonction where
where(condition, a, b) retourne a si condition est vraie, sinon b :

```python
where(x < 0, 0, 1)
```
Si seul condition est fourni, retourne les indices où la condition est vraie :

```python
where(b > 5)
```

5.4 Performance et vectorisation
Python est interprété donc plus lent que le C ou le FORTRAN.

NumPy utilise du code compilé, ce qui rend ses opérations très rapides.

Exemple :

```python
numpy.dot(a, b)  # > 100x plus rapide qu’un produit scalaire en boucle
```
5.4.1 Vectorisation
Remplace les boucles lentes par des opérations NumPy :

```python
# Lent
for i in range(len(v)):
    w[i] = v[i] + 5

# Rapide
w = v + 5
Matrice 2D – moyenne des voisins :
```

```python
A[1:-1,1:-1] = (A[:-2,1:-1] + A[2:,1:-1] +
                A[1:-1,:-2] + A[1:-1,2:]) / 4
```
np.vectorize() permet d’appliquer des fonctions personnalisées élément par élément de manière élégante.

5.5 Broadcasting (Diffusion)
La diffusion permet à NumPy d’effectuer des opérations sur des tableaux de formes différentes en les étendant automatiquement.

5.5.1 Vue mathématique
Traiter les scalaires ou tableaux de dimension inférieure comme des fonctions constantes étendues.

Exemples :

```python
v + 1  # le scalaire 1 est diffusé sur chaque élément de v
```
Addition d’une ligne et d’une colonne :

```python
C = np.arange(2).reshape(-1,1)  # forme (2,1)
R = np.arange(2).reshape(1,-1)  # forme (1,2)
C + R  # forme (2,2)

```

5.5.2 Diffusion automatique
Règles :

Ajoute des dimensions 1 à gauche si nécessaire

Étend les dimensions de taille 1 pour correspondre

Exemple réussi :

```python
(4, 3) + (3,) → broadcasted à (4, 3)
```
Exemple échoué :

```python
(3, 4) + (3,) → erreur
```
Solution :

```python
(3,) → reshape en (3,1) → résultat (3,4)
```
5.5.3 Exemples typiques
Redimensionner les lignes :

```python

rescaled = M * coeff.reshape(-1, 1)
```
Redimensionner les colonnes :

```python

rescaled = M * coeff.reshape(1, -1)
```
Fonctions de deux variables :

```python
W = u.reshape(-1, 1) + v
```
Utiliser ogrid :

```python
x, y = ogrid[0:1:3j, 0:1:3j]
w = cos(x) + sin(2*y)
```
5.6 Matrices creuses (sparse)
Utilisées pour gagner de la mémoire et améliorer les performances avec de grandes matrices comportant peu de valeurs non nulles.

Utiliser scipy.sparse.

5.6.1 Formats courants
CSR (Compressed Sparse Row) – optimisé pour les lignes

CSC (Compressed Sparse Column) – optimisé pour les colonnes

LIL (List of Lists) – idéal pour la construction et modification

5.6.2 Génération
Fonctions utiles :

```python
sp.eye(), sp.identity(), sp.spdiags(), sp.rand()
sp.csr_matrix((m, n))  # matrice nulle creuse
```
5.6.3 Méthodes
Convertir : .tocsr(), .tocsc(), .tolil()

Conversion complète : .toarray()

Multiplication : .dot()

Les opérations élémentaires (+, *, etc.) retournent un CSR

Pour des fonctions personnalisées : appliquer sur .data

5.7 Résumé
Comprendre les vues, indexations booléennes et broadcasting permet d’écrire du code NumPy rapide et élégant.

Les matrices creuses sont cruciales pour les grands problèmes scientifiques et sont gérées efficacement par scipy.sparse.