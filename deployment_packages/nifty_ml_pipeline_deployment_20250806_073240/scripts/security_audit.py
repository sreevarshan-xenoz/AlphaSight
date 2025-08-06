#!/usr/bin/env python3
"""
Security audit script for NIFTY 50 ML Pipeline.

This script conducts a comprehensive security review including:
- Dependency vulnerability scanning
- Code security analysis
- Configuration security validation
- API key and secret management review
- File permission checks
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class SecurityFinding:
    """Security audit finding."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # DEPENDENCY, CODE, CONFIG, PERMISSION, SECRET
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    cve_id: Optional[str] = None


@dataclass
class SecurityReport:
    """Complete security audit report."""
    timestamp: datetime
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    findings: List[SecurityFinding]
    summary: Dict[str, Any]


class SecurityAuditor:
    """Comprehensive security auditor for the ML pipeline."""
    
    def __init__(self, project_root: Path):
        """Initialize security auditor.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.findings: List[SecurityFinding] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Security patterns to check
        self.secret_patterns = [
            (r'api[_-]?key\s*[=:]\s*["\']([^"\']+)["\']', 'API Key'),
            (r'secret[_-]?key\s*[=:]\s*["\']([^"\']+)["\']', 'Secret Key'),
            (r'password\s*[=:]\s*["\']([^"\']+)["\']', 'Password'),
            (r'token\s*[=:]\s*["\']([^"\']+)["\']', 'Token'),
            (r'aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*["\']([^"\']+)["\']', 'AWS Access Key'),
            (r'aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']([^"\']+)["\']', 'AWS Secret Key'),
        ]
        
        self.insecure_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', 'Shell injection risk'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'pickle\.loads?\s*\(', 'Unsafe pickle usage'),
            (r'yaml\.load\s*\((?!.*Loader)', 'Unsafe YAML loading'),
            (r'requests\.get\s*\(.*verify\s*=\s*False', 'SSL verification disabled'),
        ]
    
    def add_finding(self, severity: str, category: str, title: str, 
                   description: str, file_path: Optional[str] = None,
                   line_number: Optional[int] = None, 
                   recommendation: Optional[str] = None,
                   cve_id: Optional[str] = None) -> None:
        """Add a security finding.
        
        Args:
            severity: Severity level
            category: Finding category
            title: Finding title
            description: Detailed description
            file_path: Optional file path
            line_number: Optional line number
            recommendation: Optional recommendation
            cve_id: Optional CVE identifier
        """
        finding = SecurityFinding(
            severity=severity,
            category=category,
            title=title,
            description=description,
            file_path=file_path,
            line_number=line_number,
            recommendation=recommendation,
            cve_id=cve_id
        )
        self.findings.append(finding)
        
        self.logger.info(f"Security finding: {severity} - {title}")
    
    def scan_dependencies(self) -> None:
        """Scan dependencies for known vulnerabilities."""
        self.logger.info("Scanning dependencies for vulnerabilities")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.add_finding(
                "MEDIUM", "DEPENDENCY", "Missing requirements.txt",
                "No requirements.txt file found for dependency tracking",
                recommendation="Create requirements.txt with pinned versions"
            )
            return
        
        # Check for safety tool
        try:
            result = subprocess.run(
                ["safety", "check", "-r", str(requirements_file), "--json"],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # Parse safety output
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        self.add_finding(
                            "HIGH", "DEPENDENCY", 
                            f"Vulnerable dependency: {vuln.get('package', 'Unknown')}",
                            f"Package {vuln.get('package')} version {vuln.get('installed_version')} "
                            f"has vulnerability: {vuln.get('vulnerability', 'Unknown')}",
                            recommendation=f"Update to version {vuln.get('safe_version', 'latest')}",
                            cve_id=vuln.get('cve')
                        )
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse safety output")
            else:
                self.logger.warning(f"Safety check failed: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.add_finding(
                "INFO", "DEPENDENCY", "Safety tool not available",
                "Could not run dependency vulnerability scan with safety tool",
                recommendation="Install safety tool: pip install safety"
            )
        
        # Check for unpinned dependencies
        with open(requirements_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' not in line and '>=' not in line and '<=' not in line:
                        self.add_finding(
                            "MEDIUM", "DEPENDENCY", "Unpinned dependency",
                            f"Dependency '{line}' is not pinned to a specific version",
                            file_path=str(requirements_file),
                            line_number=line_num,
                            recommendation="Pin to specific version for reproducible builds"
                        )
    
    def scan_code_security(self) -> None:
        """Scan code for security vulnerabilities."""
        self.logger.info("Scanning code for security issues")
        
        # Get all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip virtual environment and cache directories
            if any(part in str(file_path) for part in ['.venv', '__pycache__', '.git']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check for hardcoded secrets
                for line_num, line in enumerate(lines, 1):
                    for pattern, secret_type in self.secret_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            secret_value = match.group(1) if match.groups() else match.group(0)
                            # Skip obvious placeholders
                            if secret_value.lower() not in ['your_key_here', 'placeholder', 'example', 'test']:
                                self.add_finding(
                                    "CRITICAL", "SECRET", f"Hardcoded {secret_type}",
                                    f"Potential hardcoded {secret_type.lower()} found in code",
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=line_num,
                                    recommendation="Move secrets to environment variables or secure vault"
                                )
                
                # Check for insecure code patterns
                for line_num, line in enumerate(lines, 1):
                    for pattern, issue_type in self.insecure_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            self.add_finding(
                                "HIGH", "CODE", issue_type,
                                f"Potentially insecure code pattern detected",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                recommendation="Review and secure this code pattern"
                            )
                
                # Check for SQL injection risks (basic check)
                if re.search(r'execute\s*\(.*%.*\)', content, re.IGNORECASE):
                    self.add_finding(
                        "HIGH", "CODE", "Potential SQL injection",
                        "String formatting in SQL execute detected",
                        file_path=str(file_path.relative_to(self.project_root)),
                        recommendation="Use parameterized queries"
                    )
                
            except (UnicodeDecodeError, PermissionError) as e:
                self.logger.warning(f"Could not read file {file_path}: {e}")
    
    def scan_configuration_security(self) -> None:
        """Scan configuration files for security issues."""
        self.logger.info("Scanning configuration security")
        
        # Check .env files
        env_files = list(self.project_root.rglob(".env*"))
        for env_file in env_files:
            if env_file.name == '.env.example':
                continue  # Skip example files
            
            try:
                with open(env_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            # Check for suspicious values
                            if value and value not in ['', 'your_key_here', 'placeholder']:
                                for pattern, secret_type in self.secret_patterns:
                                    if re.search(pattern, line, re.IGNORECASE):
                                        self.add_finding(
                                            "HIGH", "CONFIG", f"Exposed {secret_type} in .env",
                                            f"Potential {secret_type.lower()} found in environment file",
                                            file_path=str(env_file.relative_to(self.project_root)),
                                            line_number=line_num,
                                            recommendation="Ensure .env files are in .gitignore"
                                        )
            except (UnicodeDecodeError, PermissionError) as e:
                self.logger.warning(f"Could not read env file {env_file}: {e}")
        
        # Check if .env is in .gitignore
        gitignore_file = self.project_root / ".gitignore"
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                gitignore_content = f.read()
                if '.env' not in gitignore_content:
                    self.add_finding(
                        "MEDIUM", "CONFIG", ".env not in .gitignore",
                        "Environment files may be committed to version control",
                        file_path=".gitignore",
                        recommendation="Add .env to .gitignore"
                    )
        
        # Check configuration files for insecure settings
        config_files = list(self.project_root.rglob("config/*.py"))
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                    # Check for debug mode in production
                    if re.search(r'debug\s*=\s*True', content, re.IGNORECASE):
                        self.add_finding(
                            "MEDIUM", "CONFIG", "Debug mode enabled",
                            "Debug mode may be enabled in configuration",
                            file_path=str(config_file.relative_to(self.project_root)),
                            recommendation="Ensure debug mode is disabled in production"
                        )
                    
                    # Check for insecure SSL settings
                    if re.search(r'verify\s*=\s*False', content, re.IGNORECASE):
                        self.add_finding(
                            "HIGH", "CONFIG", "SSL verification disabled",
                            "SSL certificate verification is disabled",
                            file_path=str(config_file.relative_to(self.project_root)),
                            recommendation="Enable SSL verification for security"
                        )
                        
            except (UnicodeDecodeError, PermissionError) as e:
                self.logger.warning(f"Could not read config file {config_file}: {e}")
    
    def check_file_permissions(self) -> None:
        """Check file permissions for security issues."""
        self.logger.info("Checking file permissions")
        
        # Check for world-writable files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check if world-writable (others have write permission)
                    if mode & 0o002:
                        self.add_finding(
                            "MEDIUM", "PERMISSION", "World-writable file",
                            f"File is writable by all users",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation="Restrict file permissions"
                        )
                    
                    # Check for executable files that shouldn't be
                    if file_path.suffix in ['.py', '.json', '.txt', '.md'] and mode & 0o111:
                        if not str(file_path).startswith('scripts/'):
                            self.add_finding(
                                "LOW", "PERMISSION", "Unnecessary executable permission",
                                f"Non-script file has executable permission",
                                file_path=str(file_path.relative_to(self.project_root)),
                                recommendation="Remove executable permission"
                            )
                            
                except (OSError, PermissionError):
                    continue
    
    def check_api_security(self) -> None:
        """Check API security configurations."""
        self.logger.info("Checking API security")
        
        # Check for API key management
        config_files = list(self.project_root.rglob("config/*.py"))
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                    # Check for API keys in code
                    if re.search(r'api[_-]?key.*=.*["\'][^"\']+["\']', content, re.IGNORECASE):
                        self.add_finding(
                            "HIGH", "SECRET", "API key in configuration",
                            "API key appears to be hardcoded in configuration",
                            file_path=str(config_file.relative_to(self.project_root)),
                            recommendation="Use environment variables for API keys"
                        )
                    
                    # Check for rate limiting
                    if 'rate_limit' not in content.lower():
                        self.add_finding(
                            "MEDIUM", "CONFIG", "No rate limiting configuration",
                            "API rate limiting configuration not found",
                            file_path=str(config_file.relative_to(self.project_root)),
                            recommendation="Implement API rate limiting"
                        )
                        
            except (UnicodeDecodeError, PermissionError) as e:
                self.logger.warning(f"Could not read config file {config_file}: {e}")
    
    def check_logging_security(self) -> None:
        """Check logging configuration for security issues."""
        self.logger.info("Checking logging security")
        
        # Find logging configurations
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part in str(file_path) for part in ['.venv', '__pycache__', '.git']):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Check for sensitive data in logs
                    if re.search(r'log.*password', content, re.IGNORECASE):
                        self.add_finding(
                            "HIGH", "CODE", "Password in logs",
                            "Potential password logging detected",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation="Avoid logging sensitive information"
                        )
                    
                    if re.search(r'log.*api[_-]?key', content, re.IGNORECASE):
                        self.add_finding(
                            "HIGH", "CODE", "API key in logs",
                            "Potential API key logging detected",
                            file_path=str(file_path.relative_to(self.project_root)),
                            recommendation="Avoid logging API keys"
                        )
                        
            except (UnicodeDecodeError, PermissionError) as e:
                self.logger.warning(f"Could not read file {file_path}: {e}")
    
    def run_comprehensive_audit(self) -> SecurityReport:
        """Run comprehensive security audit.
        
        Returns:
            SecurityReport with all findings
        """
        self.logger.info("Starting comprehensive security audit")
        
        # Clear existing findings
        self.findings.clear()
        
        # Run all security checks
        self.scan_dependencies()
        self.scan_code_security()
        self.scan_configuration_security()
        self.check_file_permissions()
        self.check_api_security()
        self.check_logging_security()
        
        # Count findings by severity
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'INFO': 0
        }
        
        for finding in self.findings:
            severity_counts[finding.severity] += 1
        
        # Create summary
        summary = {
            'total_files_scanned': len(list(self.project_root.rglob("*.py"))),
            'categories_checked': ['DEPENDENCY', 'CODE', 'CONFIG', 'PERMISSION', 'SECRET'],
            'most_common_category': self._get_most_common_category(),
            'risk_score': self._calculate_risk_score(severity_counts),
            'recommendations_count': len([f for f in self.findings if f.recommendation])
        }
        
        report = SecurityReport(
            timestamp=datetime.now(),
            total_findings=len(self.findings),
            critical_count=severity_counts['CRITICAL'],
            high_count=severity_counts['HIGH'],
            medium_count=severity_counts['MEDIUM'],
            low_count=severity_counts['LOW'],
            info_count=severity_counts['INFO'],
            findings=self.findings,
            summary=summary
        )
        
        self.logger.info(f"Security audit completed: {len(self.findings)} findings")
        return report
    
    def _get_most_common_category(self) -> str:
        """Get the most common finding category."""
        if not self.findings:
            return "None"
        
        categories = {}
        for finding in self.findings:
            categories[finding.category] = categories.get(finding.category, 0) + 1
        
        return max(categories, key=categories.get) if categories else "None"
    
    def _calculate_risk_score(self, severity_counts: Dict[str, int]) -> int:
        """Calculate overall risk score (0-100).
        
        Args:
            severity_counts: Count of findings by severity
            
        Returns:
            Risk score from 0 (low risk) to 100 (high risk)
        """
        # Weight different severities
        weights = {
            'CRITICAL': 25,
            'HIGH': 10,
            'MEDIUM': 5,
            'LOW': 2,
            'INFO': 1
        }
        
        total_score = sum(
            severity_counts[severity] * weight 
            for severity, weight in weights.items()
        )
        
        # Cap at 100
        return min(total_score, 100)
    
    def generate_report_text(self, report: SecurityReport) -> str:
        """Generate human-readable security report.
        
        Args:
            report: Security report
            
        Returns:
            Formatted report text
        """
        lines = []
        lines.append("="*80)
        lines.append("SECURITY AUDIT REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Findings: {report.total_findings}")
        lines.append(f"Risk Score: {report.summary['risk_score']}/100")
        lines.append("")
        
        # Summary by severity
        lines.append("FINDINGS BY SEVERITY:")
        lines.append(f"  Critical: {report.critical_count}")
        lines.append(f"  High:     {report.high_count}")
        lines.append(f"  Medium:   {report.medium_count}")
        lines.append(f"  Low:      {report.low_count}")
        lines.append(f"  Info:     {report.info_count}")
        lines.append("")
        
        # Detailed findings
        if report.findings:
            lines.append("DETAILED FINDINGS:")
            lines.append("-" * 40)
            
            # Group by severity
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
                severity_findings = [f for f in report.findings if f.severity == severity]
                if severity_findings:
                    lines.append(f"\n{severity} SEVERITY:")
                    for i, finding in enumerate(severity_findings, 1):
                        lines.append(f"\n{i}. {finding.title}")
                        lines.append(f"   Category: {finding.category}")
                        lines.append(f"   Description: {finding.description}")
                        if finding.file_path:
                            location = finding.file_path
                            if finding.line_number:
                                location += f":{finding.line_number}"
                            lines.append(f"   Location: {location}")
                        if finding.recommendation:
                            lines.append(f"   Recommendation: {finding.recommendation}")
                        if finding.cve_id:
                            lines.append(f"   CVE: {finding.cve_id}")
        else:
            lines.append("No security findings detected!")
        
        return "\n".join(lines)
    
    def save_report(self, report: SecurityReport, output_dir: str = "security_reports") -> Tuple[str, str]:
        """Save security report to files.
        
        Args:
            report: Security report
            output_dir: Output directory
            
        Returns:
            Tuple of (JSON report path, text report path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = output_path / f"security_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save text report
        text_path = output_path / f"security_report_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write(self.generate_report_text(report))
        
        self.logger.info(f"Security reports saved to {json_path} and {text_path}")
        return str(json_path), str(text_path)


def main():
    """Main function to run security audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security audit for NIFTY 50 ML Pipeline")
    parser.add_argument("--output-dir", default="security_reports",
                       help="Output directory for reports")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create auditor
    auditor = SecurityAuditor(project_root)
    
    try:
        # Run comprehensive audit
        report = auditor.run_comprehensive_audit()
        
        # Save reports
        json_path, text_path = auditor.save_report(report, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("SECURITY AUDIT SUMMARY")
        print("="*60)
        print(f"Total findings: {report.total_findings}")
        print(f"Risk score: {report.summary['risk_score']}/100")
        print(f"Critical: {report.critical_count}")
        print(f"High: {report.high_count}")
        print(f"Medium: {report.medium_count}")
        print(f"Low: {report.low_count}")
        print(f"Info: {report.info_count}")
        
        if report.critical_count > 0 or report.high_count > 0:
            print(f"\n⚠️  ATTENTION: {report.critical_count + report.high_count} high-priority security issues found!")
            print("Review the detailed report and address critical/high severity findings immediately.")
        elif report.medium_count > 0:
            print(f"\n✓ Good security posture with {report.medium_count} medium-priority items to address.")
        else:
            print("\n✅ Excellent! No high-priority security issues found.")
        
        print(f"\nDetailed reports saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {text_path}")
        
        # Return appropriate exit code
        if report.critical_count > 0:
            return 2  # Critical issues
        elif report.high_count > 0:
            return 1  # High issues
        else:
            return 0  # No high-priority issues
        
    except Exception as e:
        print(f"Security audit failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())