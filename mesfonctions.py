import numpy as np
import matplotlib.pyplot as plt


def display_scree_plot(pca):
    """ Affiche le graphique des éboulis des valeurs propres 
    et le critère de Kaiser
    On utilise le code de Nicolas Rangeon (avec qques modifications)
    disponible ici :
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-
    exploratoire-de-donnees/5345201-tp-realisez-une-acp
    Args :
    - pca : sklearn.decomposition.PCA
    Returns : 
    - graphique matplotlib
    """
    scree = pca.explained_variance_ratio_ * 100
    plt.style.use('seaborn')
    plt.figure(edgecolor='black', linewidth=4)
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.axhline(100 / pca.n_features_,
                0,
                len(pca.components_) + 1,
                c='green',
                linewidth=0.6)  # critère de Kaiser
    plt.show(block=False)


def display_circles(pca,
                    axis_ranks,
                    labels=None,
                    label_rotation=0,
                    lims=None,
                    size_nom_variable=18):
    """ Affiche les cercles des corrélation pour les plan factoriels.
    On utilise le code de Nicolas Rangeon (avec qques modifications) 
    disponible ici :
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-
    exploratoire-de-donnees/5345201-tp-realisez-une-acp
    Args :
    - pca : sklearn.decomposition.PCA
    - axis_ranks : liste des plans factoriel à tracer. 
    Exemple :  [(0,1), (2,3)] pour tracer les deux premiers plans.
    - labels : liste de nom des variables.
    - label_rotation : rotation (degrés) de l'affichage des labels.
    - lims : 'auto' ou 'None' ou tuple (xmin, xmax, ymin, ymax)
    des limites du tracé.
    - size_nom_variable : taille de l'affichage  des variables projetées.
    Returns : 
    - affiche une figure matplotlib.
    """
    n_comp = pca.n_components_
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            plt.style.use('seaborn')
            plt.figure(edgecolor='black', linewidth=4, figsize=(10, 10))

            # détermination des limites du graphique
            pcs = pca.components_
            if lims == 'auto':
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(
                    pcs[d2, :]), max(pcs[d2, :])
            elif lims is not None:  # lims est un tuple
                xmin, xmax, ymin, ymax = lims
            else:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]),
                           np.zeros(pcs.shape[1]),
                           pcs[d1, :],
                           pcs[d2, :],
                           angles='xy',
                           scale_units='xy',
                           scale=1,
                           color="grey",
                           width=0.001)

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x,
                                 y,
                                 labels[i],
                                 fontsize=str(size_nom_variable),
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation,
                                 color="blue",
                                 alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(
                d1 + 1, d2 + 1))
            plt.axis('square')

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.show()


def display_factorial_planes(X_projected,
                             pca,
                             axis_ranks,
                             labels=None,
                             size=5,
                             alpha=1,
                             illustrative_var=None):
    """ Affiche les objets (i.e. les lignes du dataset) 
    sur les plan factoriels.
    On utilise le code de Nicolas Rangeon (avec qques modifications) 
    disponible ici :
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-
    exploratoire-de-donnees/5345201-tp-realisez-une-acp
    Args :
    - X_projected : array numpy contenant les données (X) après scaling et 
    projection : X_projected = pca.transform(X_scaled)
    - pca : sklearn.decomposition.PCA
    - axis_ranks : liste des plans factoriel à tracer. 
    Exemple :  [(0,1), (2,3)] pour tracer les deux premiers plans.
    - labels : liste de nom des objets.
    - alpha : alpha ("transparence").
    - illustrative_var : variable catégorielle (série pandas avec 
    même nombre d'objets que X_projected).
    Returns : 
    - affiche une figure matplotlib.
    """
    n_comp = pca.n_components_

    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            plt.style.use('seaborn')
            plt.figure(edgecolor='black', linewidth=4, figsize=(7, 6))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2],
                            alpha=alpha, s=size)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1],
                                X_projected[selected, d2],
                                alpha=alpha, s=size,
                                label=value)
                plt.legend(markerscale=5)

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x + 0.1,
                             y + 0.1,
                             labels[i],
                             fontsize='12',
                             ha='center',
                             va='center')

            # détermination des limites du graphique
            xmin = np.min(X_projected[:, d1])
            xmax = np.max(X_projected[:, d1])
            xdelta = xmax - xmin
            ymin = np.min(X_projected[:, d2])
            ymax = np.max(X_projected[:, d2])
            ydelta = ymax - ymin
            plt.xlim(xmin - 0.02 * xdelta, xmax + 0.02 * xdelta)
            plt.ylim(ymin - 0.02 * ydelta, ymax + 0.02 * ydelta)

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.title("Projection des clients (sur F{} et F{})".format(
                d1 + 1, d2 + 1))
            plt.show(block=False)