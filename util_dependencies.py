import pkg_resources
from packaging import version
from collections import defaultdict
import subprocess

class PackageDependencyChecker:
    """Utility functions for unspaghettifying version hell."""
    def __init__(self):
        self.installed_packages = {dist.project_name: dist for dist in pkg_resources.working_set}

    def find_dependents(self, package_name):
        dependents = []
        for dist in self.installed_packages.values():
            requires = [str(req) for req in dist.requires()]
            if any(package_name in req for req in requires):
                dependents.append((dist.project_name, dist.version, requires))
        return dependents

    def check_version_discrepancy_for_line(self, line: str) -> dict:
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        if '==' in line:
            package_name, required_version = line.split('==')
            return self._check_version(package_name, required_version)
        else:
            parts = line.split()
            if len(parts) == 2:
                package_name, version_spec = parts
                return self._check_version(package_name, version_spec)
            else:
                return None

    def _check_version(self, package_name, version_spec):
        if package_name in self.installed_packages:
            installed_version = self.installed_packages[package_name].version
            requirement = pkg_resources.Requirement.parse(f"{package_name} {version_spec}")
            if not requirement.specifier.contains(installed_version):
                dependents = self.find_dependents(package_name)
                return {
                    'package_name': package_name,
                    'installed_version': installed_version,
                    'required_version': version_spec,
                    'dependents': dependents
                }
        else:
            return {
                'package_name': package_name,
                'installed_version': None,
                'required_version': version_spec,
                'dependents': []
            }
        return None

    def check_version_discrepancies(self, requirements_file):
        discrepancies = []
        with open(requirements_file, 'r') as file:
            for line in file:
                discrepancy = self.check_version_discrepancy_for_line(line)
                if discrepancy:
                    discrepancies.append(discrepancy)
        return discrepancies

    def generate_user_instructions(self, discrepancies):
        instructions = []
        for discrepancy in discrepancies:
            if discrepancy['installed_version'] is None:
                instructions.append(f"Package '{discrepancy['package_name']}' is missing. Please install version {discrepancy['required_version']}.")
            else:
                instructions.append(f"Package '{discrepancy['package_name']}' has version {discrepancy['installed_version']} installed, but version {discrepancy['required_version']} is required.")
            if discrepancy['dependents']:
                instructions.append("Dependent packages:")
                for dependent in discrepancy['dependents']:
                    instructions.append(f"  - {dependent[0]} (version {dependent[1]}) requires {discrepancy['package_name']}.")
        return "\n".join(instructions)

    def full_check(self, requirements_file):
        discrepancies = self.check_version_discrepancies(requirements_file)
        instructions = self.generate_user_instructions(discrepancies)
        return instructions

    def check_dependents_discrepancies(self, package_name):
        discrepancies = []
        dependents = self.find_dependents(package_name)
        for dependent in dependents:
            for requirement in dependent[2]:
                if package_name in requirement:
                    requirement_version = requirement.split(package_name)[-1].strip()
                    if requirement_version:
                        discrepancy = self._check_version(package_name, requirement_version)
                        if discrepancy:
                            discrepancies.append(discrepancy)
        return discrepancies

    def analyze_discrepancies(self, discrepancies):
        """Analyzes discrepancies and finds possible version ranges."""
        requirements = defaultdict(list)
        for discrepancy in discrepancies:
            requirements[discrepancy['package_name']].append(discrepancy['required_version'])

        solutions = {}
        for package, version_specs in requirements.items():
            solutions[package] = self.find_version_solutions(version_specs)
        return solutions

    def find_version_solutions(self, version_specs):
        """Finds version ranges that satisfy the given version specifications."""
        ranges = []
        for spec in version_specs:
            requirement = pkg_resources.Requirement.parse(f"dummy {spec}")
            ranges.append(requirement.specifier)
        
        return self.find_common_version_range(ranges)

    def find_common_version_range(self, ranges):
        """Finds common version range for the given specifiers."""
        min_version = version.parse("0")
        max_version = version.parse("9999")
        excluded_versions = set()
        
        for r in ranges:
            for spec in r:
                if spec.operator == '>=':
                    min_version = max(min_version, version.parse(spec.version))
                elif spec.operator == '>':
                    new_min_version = version.parse(spec.version)
                    min_version = max(min_version, version.parse(f"{new_min_version.major}.{new_min_version.minor}.{new_min_version.micro}") + version.parse("0.0.1"))
                elif spec.operator == '<=':
                    max_version = min(max_version, version.parse(spec.version))
                elif spec.operator == '<':
                    max_version = min(max_version, version.parse(spec.version))
                elif spec.operator == '!=':
                    excluded_versions.add(version.parse(spec.version))
        
        if min_version <= max_version:
            return (min_version, max_version, excluded_versions)
        else:
            return None

    def fetch_available_versions(self, package_name):
        """Fetches all available versions of a package from pip."""
        result = subprocess.run(['pip', 'index', 'versions', package_name], capture_output=True, text=True)
        if result.returncode != 0:
            return []

        lines = result.stdout.splitlines()
        versions = []
        for line in lines:
            if line.startswith('Available versions:'):
                versions = [ver.strip() for ver in line.split(':')[1].split(',')]
                break
        return versions

    def suggest_solutions(self, solutions):
        """Generates solution suggestions based on the analyzed discrepancies."""
        suggestions = []
        for package, version_range in solutions.items():
            if version_range:
                min_version, max_version, excluded_versions = version_range
                excluded_versions_str = ", ".join(str(v) for v in sorted(excluded_versions))
                if excluded_versions:
                    suggestions.append(f"Package '{package}' can be installed in the version range {min_version} - {max_version}, excluding versions: {excluded_versions_str}.")
                else:
                    suggestions.append(f"Package '{package}' can be installed in the version range {min_version} - {max_version}.")
                
                # Fetch available versions from pip
                available_versions = self.fetch_available_versions(package)
                valid_versions = [
                    version.parse(ver) for ver in available_versions 
                    if min_version <= version.parse(ver) <= max_version and version.parse(ver) not in excluded_versions
                ]

                # Filter to keep only the latest version in each subversion
                latest_versions = self.filter_latest_versions(valid_versions)

                if latest_versions:
                    suggestions.append("Possible Commands:")
                    for ver in sorted(latest_versions, reverse=True):
                        suggestions.append(f"  - pip install {package}=={ver}")
                else:
                    suggestions.append(f"No valid versions found for '{package}' within the specified range.")
            else:
                suggestions.append(f"No common version range found for package '{package}'.")
        return suggestions

    def filter_latest_versions(self, versions):
        """Filters the versions to keep only the latest version in each subversion."""
        latest_versions = {}
        for ver in versions:
            major_minor = (ver.major, ver.minor)
            if major_minor not in latest_versions or ver > latest_versions[major_minor]:
                latest_versions[major_minor] = ver
        return latest_versions.values()


if __name__ == "__main__":
    import sys
    # Usage example:
    checker = PackageDependencyChecker()

    # Find dependents of a specific package
    package_name = sys.argv[1]
    dependents = checker.find_dependents(package_name)
    print(f"Dependents of '{package_name}':")
    for dependent in dependents:
        print(f"Package: {dependent[0]}, Version: {dependent[1]}")
        print(f"Requires: {dependent[2]}")

    # Check version discrepancies based on requirements.txt
    # discrepancies = checker.check_version_discrepancies('requirements.txt')
    # print("\nVersion discrepancies:")
    # for discrepancy in discrepancies:
    #     print(f"Package: {discrepancy['package_name']}, Installed: {discrepancy['installed_version']}, Required: {discrepancy['required_version']}")
    #     print("Dependents:")
    #     for dependent in discrepancy['dependents']:
    #         print(f"  - Dependent Package: {dependent[0]}, Version: {dependent[1]}")
    #         print(f"    Requires: {dependent[2]}")

    # instructions = checker.generate_user_instructions(discrepancies)
    # print("\nVersion discrepancies and instructions:")
    # print(instructions)

    # Check version discrepancies among dependents of the specific package
    dependent_discrepancies = checker.check_dependents_discrepancies(package_name)
    if dependent_discrepancies:
        print(f"\nDiscrepancies in dependents of '{package_name}':")
        for discrepancy in dependent_discrepancies:
            print(f"Package: {discrepancy['package_name']}, Installed: {discrepancy['installed_version']}, Required: {discrepancy['required_version']}")
            print("Dependents:")
            for dependent in discrepancy['dependents']:
                print(f"  - Dependent Package: {dependent[0]}, Version: {dependent[1]}")
                print(f"    Requires: {dependent[2]}")
    else:
        print(f"No discrepancies found among the dependents of '{package_name}'.")

    # Analyze discrepancies and suggest solutions
    solutions = checker.analyze_discrepancies(dependent_discrepancies)
    solution_suggestions = checker.suggest_solutions(solutions)
    print("\nSuggested solutions:")
    for suggestion in solution_suggestions:
        print(suggestion)
