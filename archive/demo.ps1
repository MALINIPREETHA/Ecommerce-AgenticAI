# ---------------------------------------------
# DEMO SCRIPT FOR E-COMMERCE AI AGENT
# ---------------------------------------------
Write-Host "=== DEMO START: E-COMMERCE AI AGENT ===" -ForegroundColor Cyan

# 1. Call /ask endpoint (normal question)
Write-Host "`n[1] Testing /ask endpoint with question: 'What is my total sales?'" -ForegroundColor Yellow
$askResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST `
 -Headers @{"Content-Type"="application/json"} `
 -Body '{"question": "What is my total sales?"}'
Write-Host "SQL Query Generated: $($askResponse.sql_query)" -ForegroundColor Green
Write-Host "Results:" -ForegroundColor Green
$askResponse.results | Format-Table

# Save chart if present
if ($askResponse.chart_base64) {
    $base64Data = $askResponse.chart_base64 -replace "^data:image\/png;base64,", ""
    [IO.File]::WriteAllBytes("ask_chart.png", [Convert]::FromBase64String($base64Data))
    Write-Host "Chart saved as 'ask_chart.png'" -ForegroundColor Green
}

# 2. Call /visualize endpoint
Write-Host "`n[2] Testing /visualize endpoint for: 'Show total sales by item'" -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://127.0.0.1:8000/visualize?question=Show total sales by item" -Method GET -OutFile "visual_chart.png"
Write-Host "Visualization chart saved as 'visual_chart.png'" -ForegroundColor Green

# 3. Call /ask_stream endpoint (live typing effect)
Write-Host "`n[3] Testing /ask_stream endpoint with live typing effect..." -ForegroundColor Yellow
cmd /c "curl.exe -N -X POST http://127.0.0.1:8000/ask_stream -H ""Content-Type: application/json"" -d '{""question"": ""What is my total sales?""}'"

Write-Host "`n=== DEMO END ===" -ForegroundColor Cyan
