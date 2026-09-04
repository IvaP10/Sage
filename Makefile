.PHONY: protocol format check test lint verify build build-macos run-macos package-macos package-windows clean

protocol:
	./scripts/generate-protocol.sh

format:
	cargo fmt --all

check:
	cargo check --workspace --all-targets
	swift build --package-path apps/macos

test:
	cargo test --workspace

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

verify: protocol lint test check
	./scripts/check-repository.sh

build:
	cargo build --release --workspace

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
