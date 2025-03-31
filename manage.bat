@echo off
chcp 65001
title QLoRA LLM 챗봇 관리 도구

:menu
cls
echo ===================================================
echo            QLoRA LLM 챗봇 관리 도구
echo ===================================================
echo.
echo  [1] 전체 애플리케이션 실행 (백엔드 + 프론트엔드)
echo  [2] 백엔드만 실행
echo  [3] 프론트엔드만 실행
echo  [4] 프론트엔드 패키지 설치
echo  [5] 프록시 패키지 설치
echo  [6] React 앱 초기화 (주의: 기존 코드 삭제됨)
echo  [7] 실행 방법 안내
echo  [8] 종료
echo.
echo ===================================================

set /p choice=원하는 작업 번호를 선택하세요: 

if "%choice%"=="1" goto start_all
if "%choice%"=="2" goto start_backend
if "%choice%"=="3" goto start_frontend
if "%choice%"=="4" goto install_packages
if "%choice%"=="5" goto install_proxy
if "%choice%"=="6" goto setup_react
if "%choice%"=="7" goto show_help
if "%choice%"=="8" goto end

echo 올바른 번호를 입력하세요.
timeout /t 2 >nul
goto menu

:start_all
echo.
echo === QLoRA LLM 챗봇 애플리케이션 시작 ===
echo.
echo 백엔드와 프론트엔드를 함께 실행합니다.
echo.
echo 1. 백엔드 서버 시작...
start cmd /k "cd %~dp0\backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
echo 2. 백엔드 서버 시작 및 모델 로딩 중...
echo    API가 준비될 때까지 기다리는 중...

REM 백엔드 API가 준비되었는지 확인하는 루프
:check_backend_ready
timeout /t 5 > nul
echo    API 상태 확인 중...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8000/api/health' -Method Get -UseBasicParsing; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %errorlevel% neq 0 (
    echo    백엔드 API가 아직 준비되지 않았습니다. 계속 기다립니다...
    goto check_backend_ready
)

REM 백엔드가 준비되었으면 계속 진행
echo    백엔드 API가 준비되었습니다!
echo 3. 프론트엔드 개발 서버 시작...
cd %~dp0\frontend
echo 현재 위치: %CD%
echo 프론트엔드 서버를 시작합니다...
start cmd /k "npm start"
cd %~dp0
echo.
echo 모든 서버가 실행되었습니다.
echo 브라우저에서 http://localhost:3000 으로 접속하세요.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:start_backend
echo.
echo === QLoRA LLM 챗봇 백엔드 서버 시작 ===
echo.
echo 백엔드 서버만 실행합니다...
cd %~dp0\backend
start cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port 8000"
echo.
echo 백엔드 서버가 실행되었습니다.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:start_frontend
echo.
echo === QLoRA LLM 챗봇 프론트엔드 서버 시작 ===
echo.
echo 프론트엔드 서버만 실행합니다...
echo 참고: 백엔드 서버가 이미 실행 중이어야 합니다.
echo.

REM 백엔드 서버가 실행 중인지 확인
echo 백엔드 서버 연결 확인 중...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8000/api/health' -Method Get -UseBasicParsing; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %errorlevel% neq 0 (
    echo 경고: 백엔드 서버에 연결할 수 없습니다!
    echo 백엔드 서버가 실행 중인지 확인하세요.
    echo 백엔드 없이 계속 진행하시겠습니까? (Y/N)
    set /p confirm=
    if /i not "%confirm%"=="Y" goto menu
)

cd %~dp0\frontend
echo 현재 위치: %CD%
echo 프론트엔드 서버를 시작합니다...
start cmd /k "npm start"
cd %~dp0
echo.
echo 프론트엔드 서버가 실행되었습니다.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:install_packages
echo.
echo === 프론트엔드 패키지 설치 ===
echo.
cd %~dp0\frontend
echo Installing required packages...
call npm install @mui/material @mui/icons-material @emotion/react @emotion/styled react-markdown axios --force
echo.
echo 패키지 설치가 완료되었습니다.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:install_proxy
echo.
echo === 프록시 패키지 설치 ===
echo.
cd %~dp0\frontend
echo Installing http-proxy-middleware...
call npm install http-proxy-middleware --save
echo.
echo 프록시 패키지 설치가 완료되었습니다.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:setup_react
echo.
echo === React 앱 초기화 ===
echo.
echo 주의: 이 작업은 기존 프론트엔드 코드를 삭제하고 새로 생성합니다.
echo 계속하시겠습니까? (Y/N)
set /p confirm=
if /i not "%confirm%"=="Y" goto menu

echo.
echo 1. 기존 프론트엔드 디렉토리 삭제 중...
if exist "%~dp0\frontend" rmdir /s /q "%~dp0\frontend"
echo 2. React 앱 생성 중...
npx create-react-app frontend
echo 3. 필요한 패키지 설치 중...
cd "%~dp0\frontend"
call npm install @mui/material @mui/icons-material @emotion/react @emotion/styled react-markdown axios http-proxy-middleware --force
echo 4. 설정 파일 생성 중...
cd "%~dp0"

echo React 앱 초기화가 완료되었습니다.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:show_help
echo.
echo === QLoRA LLM 챗봇 실행 방법 안내 ===
echo.
echo 이 프로젝트는 다음과 같은 방법으로 실행할 수 있습니다:
echo.
echo 1. 관리 도구 사용 (현재 도구)
echo    - manage.bat 파일을 실행하여 메뉴에서 원하는 작업 선택
echo.
echo 2. 직접 실행
echo    - start.bat: 백엔드 모델 로딩 후 자동으로 프론트엔드 실행
echo    - start_frontend_only.bat: 프론트엔드만 별도로 실행
echo.
echo 3. PowerShell에서 실행
echo    - .\manage.bat - 관리 도구 실행
echo    - .\start.bat - 전체 애플리케이션 자동 실행
echo    - .\start_frontend_only.bat - 프론트엔드만 실행
echo.
echo 주의: PowerShell에서는 현재 디렉토리의 스크립트 실행을 위해
echo       반드시 '.\' 접두사를 사용해야 합니다.
echo.
echo 메인 메뉴로 돌아가려면 아무 키나 누르세요...
pause >nul
goto menu

:end
echo.
echo 프로그램을 종료합니다...
timeout /t 2 >nul
exit 