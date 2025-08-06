"""
Alerting system for production deployment monitoring.

This module provides email and Slack alerting capabilities for monitoring
pipeline performance, errors, and system health in production.
"""

import smtplib
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import requests
from enum import Enum


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories."""
    PERFORMANCE = "performance"
    ERROR = "error"
    SECURITY = "security"
    SYSTEM = "system"
    DATA = "data"


@dataclass
class Alert:
    """Alert message structure."""
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    correlation_id: Optional[str] = None


class EmailAlerter:
    """Email alerting system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize email alerter.
        
        Args:
            config: Email configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if self.enabled:
            self.smtp_server = config['smtp_server']
            self.smtp_port = config['smtp_port']
            self.username = config['smtp_username']
            self.password = config['smtp_password']
            self.from_email = config['from_email']
            self.to_emails = config['to_emails']
            self.subject_prefix = config.get('subject_prefix', '[NIFTY-PIPELINE]')
            
            logger.info(f"Email alerter initialized for {len(self.to_emails)} recipients")
        else:
            logger.info("Email alerter disabled")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Email alerter disabled, skipping alert")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"{self.subject_prefix} {alert.severity.value.upper()}: {alert.title}"
            
            # Create email body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as HTML email body.
        
        Args:
            alert: Alert to format
            
        Returns:
            HTML formatted email body
        """
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#721c24"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 20px; margin-bottom: 20px;">
                <h2 style="color: {color}; margin-top: 0;">
                    {alert.severity.value.upper()}: {alert.title}
                </h2>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Category:</strong> {alert.category.value.title()}</p>
                {f'<p><strong>Source:</strong> {alert.source}</p>' if alert.source else ''}
                {f'<p><strong>Correlation ID:</strong> {alert.correlation_id}</p>' if alert.correlation_id else ''}
            </div>
            
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">Message</h3>
                <p>{alert.message}</p>
            </div>
        """
        
        if alert.details:
            html += """
            <div style="background-color: #e9ecef; padding: 15px; border-radius: 5px;">
                <h3 style="margin-top: 0;">Details</h3>
                <pre style="background-color: white; padding: 10px; border-radius: 3px; overflow-x: auto;">
            """
            html += json.dumps(alert.details, indent=2, default=str)
            html += """
                </pre>
            </div>
            """
        
        html += """
            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px;">
                <p>This alert was generated by the NIFTY 50 ML Pipeline monitoring system.</p>
            </div>
        </body>
        </html>
        """
        
        return html


class SlackAlerter:
    """Slack alerting system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Slack alerter.
        
        Args:
            config: Slack configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if self.enabled:
            self.webhook_url = config['webhook_url']
            self.channel = config.get('channel', '#alerts')
            self.username = config.get('username', 'NIFTY Pipeline')
            
            logger.info(f"Slack alerter initialized for channel {self.channel}")
        else:
            logger.info("Slack alerter disabled")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Slack.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Slack alerter disabled, skipping alert")
            return False
        
        try:
            # Create Slack message
            payload = self._format_slack_message(alert)
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _format_slack_message(self, alert: Alert) -> Dict[str, Any]:
        """Format alert as Slack message.
        
        Args:
            alert: Alert to format
            
        Returns:
            Slack message payload
        """
        # Severity colors and emojis
        severity_config = {
            AlertSeverity.INFO: {"color": "#17a2b8", "emoji": ":information_source:"},
            AlertSeverity.WARNING: {"color": "#ffc107", "emoji": ":warning:"},
            AlertSeverity.ERROR: {"color": "#dc3545", "emoji": ":x:"},
            AlertSeverity.CRITICAL: {"color": "#721c24", "emoji": ":rotating_light:"}
        }
        
        config = severity_config.get(alert.severity, {"color": "#6c757d", "emoji": ":grey_question:"})
        
        # Create attachment
        attachment = {
            "color": config["color"],
            "title": f"{config['emoji']} {alert.severity.value.upper()}: {alert.title}",
            "text": alert.message,
            "fields": [
                {
                    "title": "Time",
                    "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "short": True
                },
                {
                    "title": "Category",
                    "value": alert.category.value.title(),
                    "short": True
                }
            ],
            "footer": "NIFTY 50 ML Pipeline",
            "ts": int(alert.timestamp.timestamp())
        }
        
        # Add optional fields
        if alert.source:
            attachment["fields"].append({
                "title": "Source",
                "value": alert.source,
                "short": True
            })
        
        if alert.correlation_id:
            attachment["fields"].append({
                "title": "Correlation ID",
                "value": alert.correlation_id,
                "short": True
            })
        
        # Add details if present
        if alert.details:
            details_text = "```\n" + json.dumps(alert.details, indent=2, default=str)[:1000] + "\n```"
            attachment["fields"].append({
                "title": "Details",
                "value": details_text,
                "short": False
            })
        
        return {
            "channel": self.channel,
            "username": self.username,
            "attachments": [attachment]
        }


class AlertManager:
    """Central alert management system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize alert manager.
        
        Args:
            config: Alerting configuration
        """
        self.config = config
        self.thresholds = config.get('thresholds', {})
        
        # Initialize alerters
        self.email_alerter = EmailAlerter(config.get('email', {}))
        self.slack_alerter = SlackAlerter(config.get('slack', {}))
        
        # Alert history for deduplication
        self.recent_alerts: List[Alert] = []
        self.max_history = 100
        
        logger.info("Alert manager initialized")
    
    def send_alert(self, severity: AlertSeverity, category: AlertCategory,
                  title: str, message: str, details: Optional[Dict[str, Any]] = None,
                  source: Optional[str] = None, correlation_id: Optional[str] = None) -> bool:
        """Send alert through configured channels.
        
        Args:
            severity: Alert severity
            category: Alert category
            title: Alert title
            message: Alert message
            details: Optional additional details
            source: Optional source identifier
            correlation_id: Optional correlation ID
            
        Returns:
            True if at least one alert was sent successfully
        """
        alert = Alert(
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            title=title,
            message=message,
            details=details,
            source=source,
            correlation_id=correlation_id
        )
        
        # Check for duplicate alerts
        if self._is_duplicate_alert(alert):
            logger.debug(f"Skipping duplicate alert: {title}")
            return False
        
        # Add to history
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > self.max_history:
            self.recent_alerts.pop(0)
        
        # Send through all configured channels
        success_count = 0
        
        if self.email_alerter.send_alert(alert):
            success_count += 1
        
        if self.slack_alerter.send_alert(alert):
            success_count += 1
        
        logger.info(f"Alert sent through {success_count} channels: {title}")
        return success_count > 0
    
    def send_performance_alert(self, metric_name: str, current_value: float,
                             threshold_value: float, details: Optional[Dict[str, Any]] = None) -> bool:
        """Send performance-related alert.
        
        Args:
            metric_name: Name of the performance metric
            current_value: Current metric value
            threshold_value: Threshold that was exceeded
            details: Optional additional details
            
        Returns:
            True if alert was sent successfully
        """
        # Determine severity based on how much threshold was exceeded
        ratio = current_value / threshold_value if threshold_value > 0 else float('inf')
        
        if ratio >= 2.0:
            severity = AlertSeverity.CRITICAL
        elif ratio >= 1.5:
            severity = AlertSeverity.ERROR
        elif ratio >= 1.2:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        title = f"Performance threshold exceeded: {metric_name}"
        message = (f"Metric '{metric_name}' value {current_value} exceeds threshold {threshold_value} "
                  f"by {((ratio - 1) * 100):.1f}%")
        
        return self.send_alert(
            severity=severity,
            category=AlertCategory.PERFORMANCE,
            title=title,
            message=message,
            details=details,
            source="performance_monitor"
        )
    
    def send_error_alert(self, error_type: str, error_message: str,
                        details: Optional[Dict[str, Any]] = None,
                        correlation_id: Optional[str] = None) -> bool:
        """Send error-related alert.
        
        Args:
            error_type: Type of error
            error_message: Error message
            details: Optional additional details
            correlation_id: Optional correlation ID
            
        Returns:
            True if alert was sent successfully
        """
        # Determine severity based on error type
        critical_errors = ['SecurityError', 'DataCorruptionError', 'SystemFailure']
        high_errors = ['APIError', 'ModelError', 'ValidationError']
        
        if any(critical in error_type for critical in critical_errors):
            severity = AlertSeverity.CRITICAL
        elif any(high in error_type for high in high_errors):
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING
        
        title = f"System error: {error_type}"
        
        return self.send_alert(
            severity=severity,
            category=AlertCategory.ERROR,
            title=title,
            message=error_message,
            details=details,
            source="error_handler",
            correlation_id=correlation_id
        )
    
    def send_system_alert(self, system_metric: str, current_value: float,
                         message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Send system-related alert.
        
        Args:
            system_metric: System metric name (cpu, memory, disk, etc.)
            current_value: Current metric value
            message: Alert message
            details: Optional additional details
            
        Returns:
            True if alert was sent successfully
        """
        # Determine severity based on metric value
        if current_value >= 95:
            severity = AlertSeverity.CRITICAL
        elif current_value >= 85:
            severity = AlertSeverity.ERROR
        elif current_value >= 75:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        title = f"System resource alert: {system_metric}"
        
        return self.send_alert(
            severity=severity,
            category=AlertCategory.SYSTEM,
            title=title,
            message=message,
            details=details,
            source="system_monitor"
        )
    
    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """Check if alert is a duplicate of recent alerts.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if this is a duplicate alert
        """
        # Check last 10 alerts for duplicates within 5 minutes
        cutoff_time = alert.timestamp.timestamp() - 300  # 5 minutes
        
        for recent_alert in self.recent_alerts[-10:]:
            if (recent_alert.timestamp.timestamp() > cutoff_time and
                recent_alert.title == alert.title and
                recent_alert.severity == alert.severity and
                recent_alert.category == alert.category):
                return True
        
        return False
    
    def test_alerting(self) -> Dict[str, bool]:
        """Test all alerting channels.
        
        Returns:
            Dictionary with test results for each channel
        """
        test_alert = Alert(
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title="Alert System Test",
            message="This is a test alert to verify the alerting system is working correctly.",
            details={"test": True, "timestamp": datetime.utcnow().isoformat()},
            source="alert_manager_test"
        )
        
        results = {
            "email": self.email_alerter.send_alert(test_alert),
            "slack": self.slack_alerter.send_alert(test_alert)
        }
        
        logger.info(f"Alert system test results: {results}")
        return results
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        recent = self.recent_alerts[-limit:] if limit > 0 else self.recent_alerts
        
        return [
            {
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "category": alert.category.value,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source,
                "correlation_id": alert.correlation_id
            }
            for alert in recent
        ]