.PHONY: protocol format check test lint verify release-ready build build-macos run-macos package-macos package-windows clean

protocol:
	./scripts/generate-protocol.sh

format:
	cargo fmt --all

check:
	cargo check --workspace --all-targets --locked
	swift build --package-path apps/macos

test:
	cargo test --workspace --locked
	cargo test -p sage-core --all-targets --features qwen35-evaluation --locked
	sh scripts/test-native-control.sh
	swift test --package-path apps/macos
	node --test integrations/browser/background.test.mjs
	python3 -m unittest discover -s scripts -p 'test_*.py'

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --locked -- -D warnings

verify: protocol lint test check
	./scripts/check-repository.sh
	python3 scripts/check-v2-release.py

release-ready:
	python3 scripts/check-v2-release.py --require-ready

build:
	cargo build --release --workspace --locked

build-macos: build
	swift build --package-path apps/macos -c release

run-macos:
	SAGE_CORE_EXECUTABLE="$(CURDIR)/target/debug/sage-core" swift run --package-path apps/macos SageMac

package-macos:
	./scripts/package-macos.sh

package-windows:
	pwsh -File scripts/package-windows.ps1

clean:
	cargo clean
	swift package --package-path apps/macos clean
