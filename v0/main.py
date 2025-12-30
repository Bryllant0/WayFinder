#!/usr/bin/env python3
"""
WayFinder - Parcoursup Analytics CLI (V0).

This is a command-line interface for analyzing Parcoursup data.
It provides formation search, statistics display, and comparison features.

Features:
    - Search formations by name or establishment
    - Display detailed statistics for a selected formation
    - Compare up to 4 formations side by side

Usage:
    # Interactive mode (default)
    python main.py

    # Direct search
    python main.py search "informatique" --year 2024

    # Show help
    python main.py --help

Author: Bryan Boislève - Mizaan-Abbas Katchera - Nawfel Bouazza
Course: Introduction to Python - CentraleSupélec
Version: 0.1
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional

from search import extract_formation_stats, search_formations
from config.settings import settings

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_header(title: str) -> None:
    """
    Print a formatted header.

    Args:
        title: The header title to display.
    """
    width = 60
    print("\n" + "=" * width)
    print(f" {title.upper()}")
    print("=" * width)


def print_divider() -> None:
    """Print a horizontal divider line."""
    print("-" * 60)


def print_formation_card(formation: Dict, index: int) -> None:
    """
    Print a formatted formation card.

    Args:
        formation: Formation data dictionary.
        index: Display index number.
    """
    lib_for = formation.get('lib_for_voe_ins') or 'Formation inconnue'
    etab = formation.get('g_ea_lib_vx') or 'Établissement inconnu'
    voeux = formation.get('voe_tot') or 0
    taux = formation.get('taux_acces') or formation.get('taux_acces_ens') or 0

    print(f"\n[{index}] {lib_for}")
    print(f"    {etab}")
    print(f"    Vœux: {voeux:,} | Taux d'accès: {taux:.0f}%")


def print_stats(stats: Dict) -> None:
    """
    Print detailed statistics for a formation.

    Args:
        stats: Formation statistics dictionary.
    """
    print_header(f"Statistiques: {stats['nom'][:50]}")
    print(f"\nÉtablissement: {stats['etablissement']}")
    print(f"Académie: {stats['academie']}")

    print_divider()
    print("\nINDICATEURS PRINCIPAUX\n")
    print(f"  Taux d'accès:    {stats['taux_acces']:.1f}%")
    print(f"  Vœux totaux:     {stats['voeux_total']:,}")
    print(f"  Admis:           {stats['admis_total']:,}")
    print(f"  Capacité:        {stats['capacite']:,}")

    print_divider()
    print("\nRÉPARTITION PAR TYPE DE BAC\n")
    print(f"  Bac Général:       {stats['pct_admis_bg']:.1f}%")
    print(f"  Bac Technologique: {stats['pct_admis_bt']:.1f}%")
    print(f"  Bac Professionnel: {stats['pct_admis_bp']:.1f}%")

    print_divider()
    print("\nRÉPARTITION PAR MENTION\n")
    print(f"  Sans mention:      {stats['pct_sans_mention']:.1f}%")
    print(f"  Assez Bien:        {stats['pct_ab']:.1f}%")
    print(f"  Bien:              {stats['pct_b']:.1f}%")
    print(f"  Très Bien:         {stats['pct_tb']:.1f}%")
    print(f"  Très Bien+:        {stats['pct_tbf']:.1f}%")

    print_divider()
    print("\nAUTRES INDICATEURS\n")
    print(f"  % Boursiers:       {stats['pct_boursiers']:.1f}%")
    print(f"  % Même académie:   {stats['pct_meme_academie']:.1f}%")
    print()


def print_comparison(formations: List[Dict]) -> None:
    """
    Print a side-by-side comparison of formations.

    Args:
        formations: List of formation statistics dictionaries.
    """
    print_header(f"Comparatif ({len(formations)} formations)")

    # Column width for each formation
    col_width = 20

    # Print formation names
    print("\n" + " " * 20, end="")
    for f in formations:
        name = f['nom'][:col_width-2]
        print(f"{name:<{col_width}}", end="")
    print()

    print_divider()

    # Metrics to compare
    metrics = [
        ("Taux d'accès", "taux_acces", "%", ".1f"),
        ("Vœux", "voeux_total", "", ","),
        ("Admis", "admis_total", "", ","),
        ("Capacité", "capacite", "", ","),
        ("% Bac Général", "pct_admis_bg", "%", ".1f"),
        ("% Mention TB+", "pct_tbf", "%", ".1f"),
        ("% Boursiers", "pct_boursiers", "%", ".1f"),
    ]

    for label, key, suffix, fmt in metrics:
        print(f"{label:<20}", end="")
        for f in formations:
            value = f.get(key, 0)
            if fmt == ",":
                formatted = f"{value:>{col_width-2},}{suffix}"
            else:
                formatted = f"{value:>{col_width-2}{fmt}}{suffix}"
            print(f"{formatted:<{col_width}}", end="")
        print()

    print()


# =============================================================================
# MENU FUNCTIONS
# =============================================================================

def menu_search(year: int = 2024) -> Optional[Dict]:
    """
    Interactive search menu.

    Args:
        year: Year to search in.

    Returns:
        Optional[Dict]: Selected formation stats or None if cancelled.
    """
    print_header("Rechercher une formation")

    search_term = input("\nEntrez votre recherche (min 3 caractères): ").strip()

    if len(search_term) < 3:
        print("La recherche doit contenir au moins 3 caractères.")
        return None

    logger.info("Searching for: %s (year: %d)", search_term, year)
    print(f"\nRecherche en cours pour '{search_term}'...")

    results = search_formations(search_term, year)

    if not results:
        print("Aucun résultat trouvé.")
        return None

    print(f"\n{len(results)} résultat(s) trouvé(s)")

    # Display results (max 10)
    nb_display = min(len(results), 10)
    for idx, f in enumerate(results[:nb_display], 1):
        print_formation_card(f, idx)

    if len(results) > nb_display:
        print(f"\n... et {len(results) - nb_display} autres résultats")

    # Selection
    print_divider()
    choice = input("\nEntrez le numéro pour voir les stats (ou 'q' pour quitter): ").strip()

    if choice.lower() == 'q':
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < nb_display:
            selected = results[idx]
            stats = extract_formation_stats(selected)
            logger.info("Formation selected: %s", stats['nom'])
            return stats
        else:
            print("Numéro invalide.")
            return None
    except ValueError:
        print("Entrée invalide.")
        return None


def menu_stats(current_selection: Optional[Dict]) -> None:
    """
    Display statistics for the currently selected formation.

    Args:
        current_selection: Currently selected formation stats.
    """
    if current_selection is None:
        print("\nAucune formation sélectionnée.")
        print("   Utilisez l'option 1 pour rechercher une formation.")
        return

    print_stats(current_selection)
    input("Appuyez sur Entrée pour continuer...")


def menu_compare(comparison_list: List[Dict]) -> List[Dict]:
    """
    Comparison menu with add/remove/display options.

    Args:
        comparison_list: Current list of formations to compare.

    Returns:
        List[Dict]: Updated comparison list.
    """
    while True:
        print_header(f"Comparatif ({len(comparison_list)}/4 formations)")

        if comparison_list:
            print("\nFormations dans le comparatif:")
            for idx, f in enumerate(comparison_list, 1):
                print(f"  [{idx}] {f['nom'][:50]}")
        else:
            print("\nAucune formation dans le comparatif.")

        print("\nOptions:")
        print("  [A] Ajouter une formation (recherche)")
        print("  [R] Retirer une formation")
        print("  [C] Afficher le comparatif")
        print("  [Q] Retour au menu principal")

        choice = input("\nVotre choix: ").strip().upper()

        if choice == 'A':
            if len(comparison_list) >= 4:
                print("Maximum 4 formations dans le comparatif.")
                continue

            stats = menu_search()
            if stats:
                # Check if already in list
                if any(f['nom'] == stats['nom'] for f in comparison_list):
                    print("Cette formation est déjà dans le comparatif.")
                else:
                    comparison_list.append(stats)
                    print(f"'{stats['nom'][:40]}...' ajoutée au comparatif.")
                    logger.info("Added to comparison: %s", stats['nom'])

        elif choice == 'R':
            if not comparison_list:
                print("Le comparatif est vide.")
                continue

            try:
                idx = int(input("Numéro à retirer: ").strip()) - 1
                if 0 <= idx < len(comparison_list):
                    removed = comparison_list.pop(idx)
                    print(f"'{removed['nom'][:40]}...' retirée.")
                    logger.info("Removed from comparison: %s", removed['nom'])
                else:
                    print("Numéro invalide.")
            except ValueError:
                print("Entrée invalide.")

        elif choice == 'C':
            if len(comparison_list) < 2:
                print("Ajoutez au moins 2 formations pour comparer.")
            else:
                print_comparison(comparison_list)
                input("Appuyez sur Entrée pour continuer...")

        elif choice == 'Q':
            break

    return comparison_list


def main_menu() -> None:
    """
    Main interactive menu loop.

    Provides options for searching, viewing stats, and comparing formations.
    """
    current_selection: Optional[Dict] = None
    comparison_list: List[Dict] = []
    year = 2024

    print("\n" + "=" * 60)
    print("   WAYFINDER - Parcoursup Analytics (V0)")
    print("   Introduction to Python - CentraleSupélec")
    print("=" * 60)

    while True:
        print("\n┌─────────────────────────────────────┐")
        print("│           MENU PRINCIPAL            │")
        print("├─────────────────────────────────────┤")
        print("│  [1] Rechercher une formation       │")
        print("│  [2] Voir les statistiques          │")
        print("│  [3] Comparatif                     │")
        print("│  [4] Changer l'année ({})          │".format(year))
        print("│  [Q] Quitter                        │")
        print("└─────────────────────────────────────┘")

        if current_selection:
            print(f"\nSélection: {current_selection['nom'][:45]}...")

        choice = input("\nVotre choix: ").strip().upper()

        if choice == '1':
            result = menu_search(year)
            if result:
                current_selection = result
                print_stats(current_selection)
                input("Appuyez sur Entrée pour continuer...")

        elif choice == '2':
            menu_stats(current_selection)

        elif choice == '3':
            comparison_list = menu_compare(comparison_list)

        elif choice == '4':
            print("\nAnnées disponibles: 2024, 2023, 2022, 2021")
            try:
                new_year = int(input("Nouvelle année: ").strip())
                if new_year in [2024, 2023, 2022, 2021]:
                    year = new_year
                    print(f"Année changée: {year}")
                    logger.info("Year changed to: %d", year)
                else:
                    print("Année non disponible.")
            except ValueError:
                print("Entrée invalide.")

        elif choice == 'Q':
            print("\nAu revoir!\n")
            break


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="WayFinder - Parcoursup Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Launch interactive mode
  %(prog)s search "CPGE MPSI"   Search for formations
  %(prog)s search "droit" -y 2023  Search in specific year

Author: Bryan Boislève - Mizaan-Abbas Katchera - Nawfel Bouazza
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Search subcommand
    search_parser = subparsers.add_parser('search', help='Search formations')
    search_parser.add_argument(
        'query',
        type=str,
        help='Search query (formation or establishment name)'
    )
    search_parser.add_argument(
        '-y', '--year',
        type=int,
        default=2024,
        choices=[2024, 2023, 2022, 2021],
        help='Year to search (default: 2024)'
    )
    search_parser.add_argument(
        '-n', '--limit',
        type=int,
        default=10,
        help='Maximum number of results to display (default: 10)'
    )

    # Global options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (debug) logging'
    )

    return parser.parse_args()


def run_search_command(query: str, year: int, limit: int) -> None:
    """
    Execute the search command from CLI.

    Args:
        query: Search query string.
        year: Year to search in.
        limit: Maximum results to display.
    """
    print_header(f"Recherche: {query} ({year})")

    if len(query) < 3:
        print("La recherche doit contenir au moins 3 caractères.")
        sys.exit(1)

    logger.info("CLI search: %s (year: %d)", query, year)
    results = search_formations(query, year)

    if not results:
        print("Aucun résultat trouvé.")
        sys.exit(0)

    print(f"\n{len(results)} résultat(s) trouvé(s)\n")

    for idx, f in enumerate(results[:limit], 1):
        print_formation_card(f, idx)

    if len(results) > limit:
        print(f"\n... et {len(results) - limit} autres résultats")
        print(f"    Utilisez --limit pour en afficher plus")

    print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Main entry point for the CLI application.

    Returns:
        int: Exit code (0 for success).
    """
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    # Handle commands
    if args.command == 'search':
        run_search_command(args.query, args.year, args.limit)
    else:
        # No command = interactive mode
        main_menu()

    return 0


if __name__ == "__main__":
    sys.exit(main())
