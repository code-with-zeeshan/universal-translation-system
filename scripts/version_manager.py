#!/usr/bin/env python3
"""
Version management system for Universal Translation System.

This script manages versions across all components while maintaining compatibility.
Vocabulary packs version independently as they're data files.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import semver
from datetime import datetime


class VersionManager:
    def __init__(self, root_dir: Path = Path('.')):
        self.root_dir = root_dir
        self.config_path = root_dir / 'version-config.json'
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load version configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Version config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _save_config(self):
        """Save version configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_current_version(self, component: str) -> str:
        """Get current version of a component."""
        if component == 'core':
            return self.config['core']['version']
        return self.config['components'][component]['version']
    
    def check_compatibility(self, component1: str, version1: str, 
                          component2: str, version2: str) -> bool:
        """Check if two component versions are compatible."""
        if component1 not in self.config['components']:
            return True  # Unknown components are assumed compatible
            
        comp_config = self.config['components'][component1]
        if 'compatibleWith' not in comp_config:
            return True
            
        if component2 not in comp_config['compatibleWith']:
            return True
            
        requirement = comp_config['compatibleWith'][component2]
        return self._check_version_requirement(version2, requirement)
    
    def _check_version_requirement(self, version: str, requirement: str) -> bool:
        """Check if version satisfies requirement string.
        Supports expressions like:
        - ">=1.0.0 <2.0.0"
        - "^1.2.0" (caret) and "~1.2.3" (tilde) will be normalized to ranges
        """
        requirement = requirement.strip()
        # Normalize caret and tilde to ranges for basic support
        if requirement.startswith('^'):
            base = requirement[1:]
            try:
                v = semver.VersionInfo.parse(base)
                upper = f"{v.major + 1}.0.0"
                requirement = f">={base} <{upper}"
            except ValueError:
                pass
        elif requirement.startswith('~'):
            base = requirement[1:]
            try:
                v = semver.VersionInfo.parse(base)
                upper = f"{v.major}.{v.minor + 1}.0"
                requirement = f">={base} <{upper}"
            except ValueError:
                pass

        parts = requirement.split()
        
        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                continue
            operator = parts[i]
            req_version = parts[i + 1]
            if operator == '>=' and not semver.compare(version, req_version) >= 0:
                return False
            elif operator == '>' and not semver.compare(version, req_version) > 0:
                return False
            elif operator == '<=' and not semver.compare(version, req_version) <= 0:
                return False
            elif operator == '<' and not semver.compare(version, req_version) < 0:
                return False
            elif operator == '==' and version != req_version:
                return False
        return True
    
    def update_version(self, component: str, new_version: str, 
                      check_compatibility: bool = True) -> Dict[str, str]:
        """Update version of a component."""
        old_version = self.get_current_version(component)
        
        # Validate semantic version
        try:
            semver.VersionInfo.parse(new_version)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {new_version}")
        
        # Check compatibility with other components
        if check_compatibility:
            incompatible = []
            for other_comp, other_config in self.config['components'].items():
                if other_comp == component:
                    continue
                    
                other_version = other_config['version']
                if not self.check_compatibility(component, new_version, 
                                              other_comp, other_version):
                    incompatible.append(f"{other_comp} {other_version}")
                    
            if incompatible:
                raise ValueError(
                    f"Version {new_version} is incompatible with: {', '.join(incompatible)}"
                )
        
        # Update configuration
        if component == 'core':
            self.config['core']['version'] = new_version
        else:
            self.config['components'][component]['version'] = new_version
            
            # Update version code for Android
            if component == 'android-sdk':
                # Convert version to version code (1.2.3 -> 10203)
                parts = new_version.split('.')
                code = int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
                self.config['components'][component]['versionCode'] = code
                
            # Update build number for iOS
            elif component == 'ios-sdk':
                # Increment build number
                current_build = int(self.config['components'][component].get('buildNumber', '100'))
                self.config['components'][component]['buildNumber'] = str(current_build + 1)
        
        # Update files
        updates = self._update_component_files(component, new_version)
        
        # Save configuration
        self._save_config()
        
        return updates
    
    def _update_component_files(self, component: str, version: str) -> Dict[str, str]:
        """Update version in component files."""
        updates = {}
        
        if component == 'android-sdk':
            # Update Android SDK Gradle files in both standalone and React Native module paths
            version_code = self.config['components'][component]['versionCode']
            gradle_paths = [
                self.root_dir / 'android' / 'UniversalTranslationSDK' / 'build.gradle',
                self.root_dir / 'react-native' / 'UniversalTranslationSDK' / 'android' / 'build.gradle',
            ]
            for gradle_path in gradle_paths:
                if gradle_path.exists():
                    content = gradle_path.read_text()
                    # Update versionName
                    content = re.sub(
                        r'versionName\s+"[^"]*"',
                        f'versionName "{version}"',
                        content,
                    )
                    # Update versionCode
                    content = re.sub(
                        r'versionCode\s+\d+',
                        f'versionCode {version_code}',
                        content,
                    )
                    gradle_path.write_text(content)
                    updates[str(gradle_path.relative_to(self.root_dir)).replace('\\','/')] = version
                
        elif component == 'ios-sdk':
            # Update podspec
            podspec = self.root_dir / 'ios' / 'UniversalTranslationSDK.podspec'
            if podspec.exists():
                content = podspec.read_text()
                content = re.sub(
                    r"s\.version\s*=\s*'[^']*'",
                    f"s.version = '{version}'",
                    content
                )
                podspec.write_text(content)
                updates['ios/podspec'] = version
                
            # Update Package.swift
            package_swift = self.root_dir / 'ios' / 'UniversalTranslationSDK' / 'Package.swift'
            if package_swift.exists():
                content = package_swift.read_text()
                # Add version comment
                if '// Version:' in content:
                    content = re.sub(
                        r'// Version: [^\n]*',
                        f'// Version: {version}',
                        content
                    )
                else:
                    lines = content.split('\n')
                    lines.insert(1, f'// Version: {version}')
                    content = '\n'.join(lines)
                package_swift.write_text(content)
                updates['ios/Package.swift'] = version
                
        elif component == 'python-package':
            # Update setup.py
            setup_py = self.root_dir / 'universal-decoder-node' / 'setup.py'
            if setup_py.exists():
                content = setup_py.read_text()
                content = re.sub(
                    r'version\s*=\s*["\'][^"\']*["\']',
                    f'version="{version}"',
                    content
                )
                setup_py.write_text(content)
                updates['python/setup.py'] = version
                
            # Update pyproject.toml
            pyproject = self.root_dir / 'universal-decoder-node' / 'pyproject.toml'
            if pyproject.exists():
                content = pyproject.read_text()
                content = re.sub(
                    r'version\s*=\s*["\'][^"\']*["\']',
                    f'version = "{version}"',
                    content
                )
                pyproject.write_text(content)
                updates['python/pyproject.toml'] = version
        
        elif component == 'web-sdk':
            # Update web SDK package.json
            pkg_path = self.root_dir / 'web' / 'universal-translation-sdk' / 'package.json'
            if pkg_path.exists():
                pkg = json.loads(pkg_path.read_text(encoding='utf-8'))
                pkg['version'] = version
                pkg_path.write_text(json.dumps(pkg, indent=2) + "\n", encoding='utf-8')
                updates['web/package.json'] = version
        
        elif component == 'react-native-sdk':
            # Update RN SDK package.json (podspec reads version from package.json)
            pkg_path = self.root_dir / 'react-native' / 'UniversalTranslationSDK' / 'package.json'
            if pkg_path.exists():
                pkg = json.loads(pkg_path.read_text(encoding='utf-8'))
                pkg['version'] = version
                pkg_path.write_text(json.dumps(pkg, indent=2) + "\n", encoding='utf-8')
                updates['react-native/package.json'] = version
        
        elif component == 'root-python':
            # Update root setup.py
            setup_py = self.root_dir / 'setup.py'
            if setup_py.exists():
                content = setup_py.read_text(encoding='utf-8')
                content = re.sub(
                    r'version\s*=\s*["\'][^"\']*["\']',
                    f'version="{version}"',
                    content
                )
                setup_py.write_text(content, encoding='utf-8')
                updates['root/setup.py'] = version
        
        # Optional: update service image tags and compose default MODEL_VERSION
        if component in {'decoder', 'coordinator'}:
            # Update Kubernetes image tags to include the version
            if component == 'decoder':
                k8s_file = self.root_dir / 'kubernetes' / 'decoder-deployment.yaml'
                if k8s_file.exists():
                    content = k8s_file.read_text(encoding='utf-8')
                    content = re.sub(r'(image:\s*universal-decoder:)[^\s]+', rf'\g<1>{version}', content)
                    content = content.replace('universal-decoder:latest', f'universal-decoder:{version}')
                    k8s_file.write_text(content, encoding='utf-8')
                    updates['k8s/decoder-deployment.yaml'] = version
            if component == 'coordinator':
                k8s_file = self.root_dir / 'kubernetes' / 'coordinator-deployment.yaml'
                if k8s_file.exists():
                    content = k8s_file.read_text(encoding='utf-8')
                    content = re.sub(r'(image:\s*universal-coordinator:)[^\s]+', rf'\g<1>{version}', content)
                    content = content.replace('universal-coordinator:latest', f'universal-coordinator:{version}')
                    k8s_file.write_text(content, encoding='utf-8')
                    updates['k8s/coordinator-deployment.yaml'] = version

            # Update Helm chart appVersion and image tag if chart exists
            chart_yaml = self.root_dir / 'charts' / 'uts' / 'Chart.yaml'
            values_yaml = self.root_dir / 'charts' / 'uts' / 'values.yaml'
            try:
                if chart_yaml.exists():
                    chart_text = chart_yaml.read_text(encoding='utf-8')
                    chart_text = re.sub(r'(?m)^appVersion:\s*"?[^"]+"?$', f'appVersion: "{version}"', chart_text)
                    chart_yaml.write_text(chart_text, encoding='utf-8')
                    updates['charts/uts/Chart.yaml'] = version
                if values_yaml.exists():
                    values_text = values_yaml.read_text(encoding='utf-8')
                    section = component  # 'coordinator' or 'decoder'
                    import re as _re
                    def _update_section_tag(text: str, section_name: str, new_tag: str) -> str:
                        m = _re.search(rf'(?m)^{section_name}:\s*\n', text)
                        if not m:
                            return text
                        start = m.end()
                        next_m = _re.search(r'(?m)^\S.*:\s*\n', text[start:])
                        end = start + (next_m.start() if next_m else len(text) - start)
                        block = text[start:end]
                        img_m = _re.search(r'(?m)^\s*image:\s*\n', block)
                        if not img_m:
                            return text
                        sub_block = block[img_m.end():]
                        def repl_tag(match: _re.Match) -> str:
                            indent = match.group(1)
                            return f"{indent}tag: \"{new_tag}\""
                        sub_block_new, n = _re.subn(r'(?m)^(\s*)tag:\s*"?[^"\n]+"?$', repl_tag, sub_block, count=1)
                        if n == 0:
                            return text
                        new_block = block[:img_m.end()] + sub_block_new
                        return text[:start] + new_block + text[end:]
                    if section in {'coordinator', 'decoder'}:
                        values_text_new = _update_section_tag(values_text, section, version)
                        if values_text_new != values_text:
                            values_text = values_text_new
                            values_yaml.write_text(values_text, encoding='utf-8')
                            updates['charts/uts/values.yaml'] = version
            except Exception:
                pass
        
        # Update default MODEL_VERSION in docker-compose.yml for visibility
        compose_file = self.root_dir / 'docker-compose.yml'
        if compose_file.exists():
            content = compose_file.read_text(encoding='utf-8')
            content = re.sub(r'MODEL_VERSION=\$\{MODEL_VERSION:-[^}]+\}', f'MODEL_VERSION=${{MODEL_VERSION:-{version}}}', content)
            compose_file.write_text(content, encoding='utf-8')
            updates['docker-compose.yml'] = version
        
        return updates
    
    def validate_all_versions(self) -> Dict[str, bool]:
        """Validate all component versions for compatibility."""
        results = {}
        
        for comp1, config1 in self.config['components'].items():
            version1 = config1['version']
            compatible = True
            
            for comp2, config2 in self.config['components'].items():
                if comp1 == comp2:
                    continue
                    
                version2 = config2['version']
                if not self.check_compatibility(comp1, version1, comp2, version2):
                    compatible = False
                    results[f"{comp1} -> {comp2}"] = False
                    
            results[comp1] = compatible
            
        return results
    
    def generate_compatibility_matrix(self) -> str:
        """Generate a compatibility matrix."""
        components = list(self.config['components'].keys())
        
        # Header
        matrix = "Compatibility Matrix\n"
        matrix += "=" * 80 + "\n"
        matrix += f"{'Component':<20}"
        for comp in components:
            matrix += f"{comp:<15}"
        matrix += "\n" + "-" * 80 + "\n"
        
        # Rows
        for comp1 in components:
            version1 = self.config['components'][comp1]['version']
            matrix += f"{comp1:<20}"
            
            for comp2 in components:
                if comp1 == comp2:
                    matrix += f"{'---':<15}"
                else:
                    version2 = self.config['components'][comp2]['version']
                    compatible = self.check_compatibility(comp1, version1, comp2, version2)
                    symbol = "✓" if compatible else "✗"
                    matrix += f"{symbol} {version2:<13}"
                    
            matrix += "\n"
            
        return matrix

    def validate_files_sync(self) -> Dict[str, bool]:
        """Validate that file versions match version-config.json for key components."""
        results: Dict[str, bool] = {}
        root = self.root_dir
        ok = True

        def _get_version_from_setup(path: Path) -> Optional[str]:
            if not path.exists():
                return None
            m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', path.read_text(encoding='utf-8'))
            return m.group(1) if m else None

        def _get_version_from_pyproject(path: Path) -> Optional[str]:
            if not path.exists():
                return None
            m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', path.read_text(encoding='utf-8'))
            return m.group(1) if m else None

        def _get_version_from_package_json(path: Path) -> Optional[str]:
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding='utf-8')).get('version')
            except Exception:
                return None

        # Web SDK
        web_cfg = self.config['components'].get('web-sdk', {})
        web_expected = web_cfg.get('version')
        web_pkg = root / 'web' / 'universal-translation-sdk' / 'package.json'
        web_found = _get_version_from_package_json(web_pkg)
        results['web-sdk/package.json'] = (web_found == web_expected)
        ok = ok and results['web-sdk/package.json']

        # React Native SDK (package.json)
        rn_cfg = self.config['components'].get('react-native-sdk', {})
        rn_expected = rn_cfg.get('version')
        rn_pkg = root / 'react-native' / 'UniversalTranslationSDK' / 'package.json'
        rn_found = _get_version_from_package_json(rn_pkg)
        results['react-native/package.json'] = (rn_found == rn_expected)
        ok = ok and results['react-native/package.json']

        # RN Podspec references package["version"] (structural check)
        rn_podspec = root / 'react-native' / 'UniversalTranslationSDK' / 'universal-translation-sdk.podspec'
        if rn_podspec.exists():
            txt = rn_podspec.read_text(encoding='utf-8')
            results['react-native/podspec'] = ('s.version' in txt and 'package["version"]' in txt)
            ok = ok and results['react-native/podspec']

        # Root Python package
        root_setup = root / 'setup.py'
        root_expected = self.config['components'].get('root-python', {}).get('version')
        root_found = _get_version_from_setup(root_setup)
        if root_expected:
            results['root/setup.py'] = (root_found == root_expected)
            ok = ok and results['root/setup.py']

        # Universal Decoder Node (setup.py and pyproject.toml)
        udn_setup = root / 'universal-decoder-node' / 'setup.py'
        udn_pyproj = root / 'universal-decoder-node' / 'pyproject.toml'
        udn_expected = self.config['components'].get('python-package', {}).get('version')
        udn_found_setup = _get_version_from_setup(udn_setup)
        udn_found_pyproj = _get_version_from_pyproject(udn_pyproj)
        results['universal-decoder-node/setup.py'] = (udn_found_setup == udn_expected)
        results['universal-decoder-node/pyproject.toml'] = (udn_found_pyproj == udn_expected)
        ok = ok and results['universal-decoder-node/setup.py'] and results['universal-decoder-node/pyproject.toml']

        # iOS SDK podspec (if present)
        ios_podspec = root / 'ios' / 'UniversalTranslationSDK.podspec'
        if ios_podspec.exists():
            ios_expected = self.config['components'].get('ios-sdk', {}).get('version')
            txt = ios_podspec.read_text(encoding='utf-8')
            m = re.search(r"s\.version\s*=\s*'([^']+)'", txt)
            ios_found = m.group(1) if m else None
            if ios_expected:
                results['ios/podspec'] = (ios_found == ios_expected)
                ok = ok and results['ios/podspec']

        results['__all__'] = ok
        return results


def main():
    parser = argparse.ArgumentParser(description='Manage versions across Universal Translation System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show current versions')
    show_parser.add_argument('component', nargs='?', help='Component name')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update component version')
    update_parser.add_argument('component', help='Component name')
    update_parser.add_argument('version', help='New version')
    update_parser.add_argument('--force', action='store_true', help='Force update even if incompatible')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check compatibility')
    check_parser.add_argument('component1', help='First component')
    check_parser.add_argument('version1', help='First version')
    check_parser.add_argument('component2', help='Second component')
    check_parser.add_argument('version2', help='Second version')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate all versions')
    
    # Validate file sync command
    validate_sync_parser = subparsers.add_parser('validate-sync', help='Validate file versions match version-config.json')
    
    # Matrix command
    matrix_parser = subparsers.add_parser('matrix', help='Show compatibility matrix')
    
    # Release command
    release_parser = subparsers.add_parser('release', help='Prepare release')
    release_parser.add_argument('version', help='Release version')
    release_parser.add_argument('--components', nargs='+', help='Components to release')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        vm = VersionManager()
        
        if args.command == 'show':
            if args.component:
                version = vm.get_current_version(args.component)
                print(f"{args.component}: {version}")
            else:
                print("Current versions:")
                print(f"  Core: {vm.config['core']['version']}")
                for comp, config in vm.config['components'].items():
                    print(f"  {comp}: {config['version']}")
                    
        elif args.command == 'update':
            updates = vm.update_version(
                args.component, 
                args.version,
                check_compatibility=not args.force
            )
            print(f"Updated {args.component} to {args.version}")
            for file, version in updates.items():
                print(f"  ✓ {file}")
                
        elif args.command == 'check':
            compatible = vm.check_compatibility(
                args.component1, args.version1,
                args.component2, args.version2
            )
            if compatible:
                print(f"✓ {args.component1} {args.version1} is compatible with {args.component2} {args.version2}")
            else:
                print(f"✗ {args.component1} {args.version1} is NOT compatible with {args.component2} {args.version2}")
                sys.exit(1)
                
        elif args.command == 'validate':
            results = vm.validate_all_versions()
            all_valid = all(v for v in results.values() if isinstance(v, bool))
            
            print("Version validation:")
            for comp, valid in results.items():
                if isinstance(valid, bool):
                    symbol = "✓" if valid else "✗"
                    print(f"  {symbol} {comp}")
                    
            if not all_valid:
                print("\n✗ Some components have incompatible versions!")
                sys.exit(1)
            else:
                print("\n✓ All versions are compatible!")
        
        elif args.command == 'validate-sync':
            results = vm.validate_files_sync()
            ok = results.get('__all__', False)
            print("File version sync validation:")
            for k, v in results.items():
                if k == '__all__':
                    continue
                symbol = '✓' if v else '✗'
                print(f"  {symbol} {k}")
            if not ok:
                sys.exit(1)
            else:
                print("\n✓ All file versions are in sync with version-config.json")
                
        elif args.command == 'matrix':
            print(vm.generate_compatibility_matrix())
            
        elif args.command == 'release':
            components = args.components or list(vm.config['components'].keys())
            
            print(f"Preparing release {args.version} for: {', '.join(components)}")
            
            for comp in components:
                updates = vm.update_version(comp, args.version)
                print(f"\n{comp}:")
                for file, version in updates.items():
                    print(f"  ✓ {file}")
                    
            print(f"\n✓ Release {args.version} prepared!")
            print("Don't forget to:")
            print("  1. Update CHANGELOG.md")
            print("  2. Create git tag: git tag -a v{args.version} -m 'Release {args.version}'")
            print("  3. Push changes: git push && git push --tags")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()