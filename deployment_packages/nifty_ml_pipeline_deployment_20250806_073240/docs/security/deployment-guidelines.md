# Security Deployment Guidelines

## Pre-Deployment Security Checklist

### 1. Environment Variables
- [ ] All sensitive values moved to environment variables
- [ ] No hardcoded API keys, passwords, or secrets in code
- [ ] .env files added to .gitignore
- [ ] Production .env file secured with appropriate permissions (600)

### 2. File Permissions
- [ ] Sensitive configuration files have restricted permissions
- [ ] Log files are not world-readable
- [ ] Model files are protected from unauthorized access

### 3. API Security
- [ ] API rate limiting configured
- [ ] SSL/TLS verification enabled for all external requests
- [ ] API timeouts configured to prevent hanging requests
- [ ] API keys rotated regularly (recommended: every 90 days)

### 4. Data Security
- [ ] Sensitive data encryption enabled
- [ ] Data retention policies configured
- [ ] Backup encryption enabled
- [ ] Access logging enabled for audit trails

### 5. Network Security
- [ ] Firewall rules configured to restrict access
- [ ] Only necessary ports exposed
- [ ] VPN or private network access for management
- [ ] SSL certificates valid and up-to-date

### 6. Monitoring and Alerting
- [ ] Security event monitoring enabled
- [ ] Failed authentication alerts configured
- [ ] Unusual activity detection enabled
- [ ] Log aggregation and analysis setup

### 7. Dependency Security
- [ ] All dependencies updated to latest secure versions
- [ ] Vulnerability scanning performed
- [ ] No known CVEs in production dependencies
- [ ] Dependency integrity verification enabled

## Production Deployment Steps

1. **Environment Setup**
   ```bash
   # Create secure directories
   sudo mkdir -p /var/lib/nifty-pipeline/{data,models}
   sudo mkdir -p /var/log/nifty-pipeline
   sudo mkdir -p /var/backups/nifty-pipeline
   
   # Set appropriate permissions
   sudo chown -R nifty-user:nifty-group /var/lib/nifty-pipeline
   sudo chown -R nifty-user:nifty-group /var/log/nifty-pipeline
   sudo chmod 750 /var/lib/nifty-pipeline
   sudo chmod 750 /var/log/nifty-pipeline
   ```

2. **SSL/TLS Configuration**
   ```bash
   # Generate or install SSL certificates
   # Configure nginx/apache for HTTPS termination
   # Ensure TLS 1.2+ only
   ```

3. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp   # SSH
   sudo ufw allow 443/tcp  # HTTPS
   sudo ufw enable
   ```

4. **Environment Variables**
   ```bash
   # Copy and configure environment file
   cp .env.production.template .env
   # Edit .env with actual values
   chmod 600 .env
   ```

5. **Service Configuration**
   ```bash
   # Create systemd service file
   # Configure log rotation
   # Set up monitoring
   ```

## Security Monitoring

### Key Metrics to Monitor
- Failed authentication attempts
- Unusual API usage patterns
- High error rates
- Memory/CPU usage spikes
- Disk space usage
- Network traffic anomalies

### Alert Thresholds
- Failed logins: > 5 in 5 minutes
- API errors: > 10% error rate
- Memory usage: > 85%
- CPU usage: > 80% for > 5 minutes
- Disk usage: > 90%

## Incident Response

### Security Incident Procedures
1. **Immediate Response**
   - Isolate affected systems
   - Preserve logs and evidence
   - Notify security team
   - Document timeline

2. **Investigation**
   - Analyze logs for attack vectors
   - Identify compromised data/systems
   - Assess impact and scope
   - Collect forensic evidence

3. **Recovery**
   - Patch vulnerabilities
   - Rotate compromised credentials
   - Restore from clean backups
   - Update security controls

4. **Post-Incident**
   - Conduct lessons learned review
   - Update security procedures
   - Implement additional controls
   - Report to stakeholders

## Regular Security Tasks

### Daily
- [ ] Review security alerts
- [ ] Check system logs for anomalies
- [ ] Verify backup completion

### Weekly
- [ ] Review access logs
- [ ] Update security patches
- [ ] Test alert systems

### Monthly
- [ ] Rotate API keys
- [ ] Review user access
- [ ] Security scan dependencies
- [ ] Update security documentation

### Quarterly
- [ ] Penetration testing
- [ ] Security architecture review
- [ ] Incident response drill
- [ ] Security training update
