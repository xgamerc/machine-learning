@echo off
SetLocal EnableDelayedExpansion

SET defines=
SET assembly=ml
SET bin_dir=..\bin
SET src_dir=src

@REM FILE GATHERING:
SET files=
@REM FOR /R %%f in (*.cpp) do (
  @REM SET files=!files! %%f
@REM )

@REM @REM @ECHO !files!
SET files=%src_dir%\playground.cpp

@REM INCLUDES:
SET includes=
SET includes=%includes% -I.\ -I..\
SET includes=%includes% -I%src_dir%\

@REM for Vulkan builds:
@REM SET includes=%includes% -I%VULKAN_SDK%\Include

@REM for SFML builds:
@REM SET includes=%includes% -I..\External\ -I..\External\SFML\include

SET linker_switches=-L%bin_dir%\

Echo "Building %assembly%.exe..."
IF NOT EXIST ..\bin\ MKDIR ..\bin\

set clang_v="C:\Program Files\LLVM\bin\clang++"
SET compiler_switches=-g -std=c++23
%clang_v% %files% %compiler_switches% -o ..\bin\%assembly%.exe %defines% %includes% %linker_switches%
