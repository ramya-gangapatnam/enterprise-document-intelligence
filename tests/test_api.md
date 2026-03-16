# API Test Cases

## Health
GET /health
Expected: {"status": "ok"}

## Upload
POST /upload
File: sample_policy.txt
Expected: document indexed successfully

## Ask - Remote work
Question: How many remote work days are allowed per week?
Expected: three days per week

## Ask - Security
Question: What security controls are required for remote access?
Expected: VPN and multi-factor authentication

## Ask - PTO
Question: How many PTO days do full-time employees receive?
Expected: 15 days annually

## Ask - Unknown
Question: What is the company stock ticker?
Expected: information is not available in the provided documents