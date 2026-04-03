import os
import sys
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # 1. CMake 설치 확인
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        
        # 2. Pybind11 설치 확인 (CMakeLists.txt의 find_package를 위해 필요)
        try:
            import pybind11
            self.pybind11_cmake_path = pybind11.get_cmake_dir()
        except ImportError:
            # 빌드 시스템에 pybind11이 없으면 자동으로 설치 시도 (setup_requires가 처리하지만 안전장치)
            print("pybind11 not found. Please install it via 'pip install pybind11'")
            sys.exit(1)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # 3. CMake 인자 설정
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={self.pybind11_cmake_path}', # <--- [핵심] Pybind11 경로 주입
            '-DCMAKE_BUILD_TYPE=' + ('Debug' if self.debug else 'Release')
        ]

        build_args = ['--config', 'Debug' if self.debug else 'Release']

        # 4. 멀티 코어 빌드 설정 (-j)
        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{"Debug" if self.debug else "Release".upper()}={extdir}']
            if sys.maxsize > 2**32: cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j2'] # 코어 수에 따라 숫자 조정 가능

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # 5. CMake 실행 (Configure -> Build)
        print(f"Building extension in: {self.build_temp}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='mycpp',
    version='0.1.0',
    description='A Pybind11 extension built with CMake',
    ext_modules=[CMakeExtension('mycpp')], # 모듈 이름이 CMakeLists.txt의 target 이름과 일치해야 함
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    setup_requires=['pybind11'], # 빌드 시 pybind11 필요
    install_requires=['pybind11'], # 설치 시 pybind11 필요
)